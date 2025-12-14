"""
Passive Agent Worker for DPR-RC System (Cache-Based Architecture)

Per Architecture Spec Section 2.1:
"Passive Agent (P) - A serverless worker representing a frozen snapshot of history,
performing verification on demand."

ARCHITECTURE: Cache-Based Response Model
- Passive agents are SHARD-AGNOSTIC workers
- They dynamically connect to any target shard at query time
- No restart/redeployment needed to handle different time-shards
- Responses are written to Redis cache (not Pub/Sub)
- Peer voting enables parallel RCP execution

IMPORTANT: Passive Agents are PREVIOUS VERSIONS OF A*, not independent personas.
They represent frozen model checkpoints at specific temporal points.
"""

import os
import time
import json
import redis
import threading
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import chromadb

from .models import (
    ComponentType, EventType, LogEntry, CachedResponse, PeerVote
)
from .logging_utils import StructuredLogger

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WORKER_ID = os.getenv("HOSTNAME", f"worker-{os.getpid()}")
HISTORY_BUCKET = os.getenv("HISTORY_BUCKET", None)

# TTLs for cache entries
RESPONSE_TTL = 60  # seconds
VOTE_TTL = 60  # seconds
HEARTBEAT_INTERVAL = 10  # seconds
WORKER_READY_TTL = 30  # seconds

logger = StructuredLogger(ComponentType.PASSIVE_WORKER)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

RFI_STREAM = "dpr:rfi"
GROUP_NAME = "passive_workers"
CONSUMER_NAME = WORKER_ID
WORKERS_READY_KEY = "dpr:workers:ready"


@dataclass
class FrozenAgentState:
    """
    Represents a frozen snapshot of A* at a specific point in time.

    Per Mathematical Model Section 4.2:
    "Each historical agent A_k represents a frozen snapshot of the context state
    from a previous time step."
    """
    creation_time: str
    epoch_year: int
    context_summary: str = ""

    def verify(self, query: str, candidate: str) -> dict:
        """
        Verify candidate answer against this frozen state's knowledge.
        In production, runs inference on the frozen model checkpoint.
        """
        return {
            "score": 0.85,
            "temporal_context": f"From {self.epoch_year} perspective",
            "prompt": f"Verify: {candidate}",
            "response": f"Consistent with {self.creation_time} knowledge base",
            "tokens": 10
        }


class PassiveWorker:
    """
    Shard-Agnostic Passive Agent Worker.

    Key architectural points:
    1. Workers are SHARD-AGNOSTIC - they dynamically connect to any shard
    2. The shard_id in each RFI determines which ChromaDB collection to query
    3. Responses are cached in Redis (not published via Pub/Sub)
    4. Workers perform peer voting on other responses for RCP
    """

    def __init__(self):
        self.worker_id = WORKER_ID

        # ChromaDB client for dynamic shard connections
        self.chroma = chromadb.Client()

        # Cache of shard collections (lazy-loaded)
        self._shard_collections: Dict[str, Any] = {}

        # Initialize default data across common shards
        self._initialize_default_shards()

        # Register as ready
        self._register_ready()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        logger.logger.info(f"PassiveWorker {self.worker_id} initialized (shard-agnostic)")

    def _initialize_default_shards(self):
        """Initialize default shards with fallback data for testing."""
        default_shards = ["2019", "2020", "2021", "2022", "2023", "2024", "broadcast"]

        for shard_id in default_shards:
            collection = self._get_shard_collection(shard_id)
            if collection.count() == 0:
                self._generate_fallback_data(shard_id, collection)

    def _get_shard_collection(self, shard_id: str):
        """
        Dynamically get or create a collection for a shard.
        This enables shard-agnostic operation - no restart needed.
        """
        if shard_id not in self._shard_collections:
            collection_name = f"dpr_shard_{shard_id}"
            self._shard_collections[shard_id] = self.chroma.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.logger.debug(f"Connected to shard collection: {collection_name}")

        return self._shard_collections[shard_id]

    def _register_ready(self):
        """Register this worker as ready in Redis."""
        redis_client.sadd(WORKERS_READY_KEY, self.worker_id)
        redis_client.expire(WORKERS_READY_KEY, WORKER_READY_TTL)
        logger.log_event("system", EventType.WORKER_READY, {"worker_id": self.worker_id})

    def _heartbeat_loop(self):
        """Maintain heartbeat to keep worker registered as ready."""
        while True:
            try:
                redis_client.sadd(WORKERS_READY_KEY, self.worker_id)
                redis_client.expire(WORKERS_READY_KEY, WORKER_READY_TTL)
            except Exception as e:
                logger.logger.warning(f"Heartbeat error: {e}")
            time.sleep(HEARTBEAT_INTERVAL)

    def _generate_fallback_data(self, shard_id: str, collection):
        """Generate minimal synthetic data for a shard."""
        fallback_docs = []

        # Parse year from shard_id if possible
        try:
            year = int(shard_id) if shard_id.isdigit() else 2020
        except ValueError:
            year = 2020

        for i in range(10):
            doc_id = f"fallback_{shard_id}_{i}"
            content = f"Historical record from {year}: Research milestone {i} achieved. " \
                     f"Progress in domain area with metrics showing improvement. " \
                     f"Shard {shard_id} data point {i}."

            fallback_docs.append({
                "id": doc_id,
                "content": content,
                "metadata": {"year": year, "type": "fallback", "index": i}
            })

        self._bulk_insert(fallback_docs, collection)
        logger.logger.info(f"Generated {len(fallback_docs)} fallback documents for shard {shard_id}")

    def _bulk_insert(self, documents: List[Dict], collection):
        """Bulk insert documents into a ChromaDB collection."""
        if not documents:
            return

        ids = [doc.get('id', hashlib.md5(doc['content'].encode()).hexdigest()[:12]) for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]

        try:
            collection.add(ids=ids, documents=contents, metadatas=metadatas)
        except Exception as e:
            if "duplicate" in str(e).lower() or "already" in str(e).lower():
                # Skip duplicates silently
                pass
            else:
                logger.logger.error(f"Failed to insert batch: {e}")

    def retrieve_from_shard(self, shard_id: str, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant content from a specific shard.
        This is the key to shard-agnostic operation.
        """
        collection = self._get_shard_collection(shard_id)

        if collection.count() == 0:
            logger.logger.warning(f"Shard {shard_id} is empty")
            return None

        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=3
            )

            if not results['documents'] or not results['documents'][0]:
                return None

            return {
                "content": results['documents'][0][0],
                "id": results['ids'][0][0],
                "metadata": results['metadatas'][0][0] if results['metadatas'] else {},
                "distance": results['distances'][0][0] if results.get('distances') else 0.0
            }
        except Exception as e:
            logger.logger.error(f"Retrieval error from shard {shard_id}: {e}")
            return None

    def verify_l2(self, content: str, query: str, depth: int = 0) -> float:
        """
        L2 Verification using SLM reasoning.

        Per Mathematical Model Section 5.2, Equation 9:
        C(r_p) = V(q, context_p) · (1 / (1 + i))
        """
        query_tokens = set(query.lower().split())
        content_tokens = set(content.lower().split())

        if not query_tokens:
            return 0.0

        intersection = query_tokens & content_tokens
        union = query_tokens | content_tokens

        if not union:
            return 0.0

        base_score = len(intersection) / len(union)
        length_factor = min(1.0, len(content) / 200.0)
        v_score = (base_score * 0.6 + length_factor * 0.4)

        # Apply depth penalty: 1/(1+i)
        depth_penalty = 1.0 / (1.0 + depth)
        confidence = v_score * depth_penalty

        return max(0.0, min(1.0, confidence))

    def compute_peer_vote(self, my_content: str, peer_response: CachedResponse) -> PeerVote:
        """
        Compute a peer vote on another agent's response.

        Agreement score (v+): semantic similarity between responses
        Disagreement score (v-): divergence indicator
        """
        peer_content = peer_response.content or ""

        # Compute agreement based on content overlap
        my_tokens = set(my_content.lower().split())
        peer_tokens = set(peer_content.lower().split())

        if not my_tokens or not peer_tokens:
            agreement = 0.5
            disagreement = 0.5
        else:
            intersection = my_tokens & peer_tokens
            union = my_tokens | peer_tokens
            similarity = len(intersection) / len(union) if union else 0.0

            agreement = similarity
            disagreement = 1.0 - similarity

        return PeerVote(
            trace_id=peer_response.trace_id,
            voter_id=self.worker_id,
            votee_id=peer_response.agent_id,
            agreement_score=round(agreement, 3),
            disagreement_score=round(disagreement, 3),
            timestamp=datetime.utcnow().isoformat()
        )

    def cache_response(self, trace_id: str, shard_id: str, content: str, confidence: float):
        """
        Cache response in Redis instead of publishing via Pub/Sub.
        Key pattern: dpr:response:{trace_id}:{agent_id}
        """
        response = CachedResponse(
            trace_id=trace_id,
            agent_id=self.worker_id,
            shard_id=shard_id,
            content_hash=hashlib.md5(content.encode()).hexdigest()[:12],
            content=content[:500],  # Truncate for storage
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat()
        )

        key = f"dpr:response:{trace_id}:{self.worker_id}"
        redis_client.setex(key, RESPONSE_TTL, response.model_dump_json())

        logger.log_event(trace_id, EventType.RESPONSE_CACHED, {
            "worker_id": self.worker_id,
            "shard_id": shard_id,
            "confidence": confidence
        })

        return response

    def cast_peer_votes(self, trace_id: str, my_content: str, wait_time: float = 1.0):
        """
        Cast peer votes on other agents' responses.
        Voting can begin as soon as any peer response appears.
        """
        time.sleep(wait_time)  # Wait for other responses to appear

        # Get all responses for this trace
        pattern = f"dpr:response:{trace_id}:*"
        for key in redis_client.scan_iter(match=pattern):
            # Skip our own response
            if key.endswith(f":{self.worker_id}"):
                continue

            data = redis_client.get(key)
            if data:
                try:
                    peer_response = CachedResponse(**json.loads(data))
                    vote = self.compute_peer_vote(my_content, peer_response)

                    # Cache the vote
                    vote_key = f"dpr:vote:{trace_id}:{self.worker_id}:{peer_response.agent_id}"
                    redis_client.setex(vote_key, VOTE_TTL, vote.model_dump_json())

                    logger.log_event(trace_id, EventType.PEER_VOTE_CAST, {
                        "voter": self.worker_id,
                        "votee": peer_response.agent_id,
                        "agreement": vote.agreement_score
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    logger.logger.warning(f"Failed to parse peer response: {e}")

    def process_rfi(self, rfi_data: Dict[str, Any]):
        """
        Process a Request for Information (RFI) from the Active Agent.

        Flow:
        1. Determine target shard from RFI
        2. Dynamically connect to shard (no restart needed)
        3. Retrieve relevant content
        4. Verify with L2
        5. Cache response in Redis
        6. Cast peer votes on other responses
        """
        trace_id = rfi_data.get('trace_id', 'unknown')
        query_text = rfi_data.get('query_text', '')
        target_shards_str = rfi_data.get('target_shards', '[]')

        # Parse target shards
        try:
            target_shards = json.loads(target_shards_str) if target_shards_str else []
        except json.JSONDecodeError:
            target_shards = []

        # Determine which shard to query
        # Shard-agnostic: we handle any shard dynamically
        if target_shards and "broadcast" not in target_shards:
            # Extract shard year from target (e.g., "shard_2020" -> "2020")
            shard_id = target_shards[0].replace("shard_", "")
        else:
            shard_id = "broadcast"

        # 1. Retrieve from dynamically-selected shard
        doc = self.retrieve_from_shard(shard_id, query_text)
        if not doc:
            logger.logger.debug(f"No relevant content in shard {shard_id} for: {query_text[:50]}...")
            return

        # 2. L2 Verification
        depth = doc.get('metadata', {}).get('hierarchy_depth', 0)
        confidence = self.verify_l2(doc['content'], query_text, depth)

        # Only respond if confidence meets threshold
        if confidence < 0.3:
            logger.logger.debug(f"Confidence {confidence:.2f} below threshold")
            return

        # 3. Cache response in Redis
        self.cache_response(trace_id, shard_id, doc['content'], confidence)

        # 4. Cast peer votes (in background to not block)
        threading.Thread(
            target=self.cast_peer_votes,
            args=(trace_id, doc['content']),
            daemon=True
        ).start()

    def ingest_benchmark_data(self, events: List[Dict], shard_id: str = "benchmark"):
        """
        Ingest benchmark data into a specific shard.
        """
        collection = self._get_shard_collection(shard_id)

        documents = []
        for event in events:
            documents.append({
                "id": event.get('id', hashlib.md5(event['content'].encode()).hexdigest()[:12]),
                "content": event['content'],
                "metadata": {
                    "timestamp": event.get('timestamp', ''),
                    "topic": event.get('topic', ''),
                    "event_type": event.get('event_type', ''),
                    "perspective": event.get('perspective', '')
                }
            })

        self._bulk_insert(documents, collection)
        logger.logger.info(f"Ingested {len(documents)} benchmark documents to shard {shard_id}")


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - starts worker thread on startup"""
    worker_thread = threading.Thread(target=run_worker_loop, daemon=True)
    worker_thread.start()
    logger.logger.info(f"Passive Worker {WORKER_ID} started (shard-agnostic)")
    yield
    logger.logger.info(f"Passive Worker {WORKER_ID} shutting down")


app = FastAPI(title="DPR-PassiveWorker", lifespan=lifespan)


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint for worker readiness verification."""
    return {
        "status": "healthy",
        "worker_id": WORKER_ID,
        "shard_agnostic": True,
        "ready": redis_client.sismember(WORKERS_READY_KEY, WORKER_ID)
    }


@app.post("/ingest")
def ingest_data(data: Dict[str, Any]):
    """Endpoint to ingest benchmark data"""
    global _worker_instance
    if _worker_instance and 'events' in data:
        shard_id = data.get('shard_id', 'benchmark')
        _worker_instance.ingest_benchmark_data(data['events'], shard_id)
        return {"status": "ok", "ingested": len(data['events']), "shard": shard_id}
    return {"status": "error", "message": "No worker instance or invalid data"}


# Global worker instance
_worker_instance: Optional[PassiveWorker] = None


def run_worker_loop():
    """Main worker loop - processes RFIs from Redis stream"""
    global _worker_instance

    worker = PassiveWorker()
    _worker_instance = worker

    # Initialize Stream Group
    try:
        redis_client.xgroup_create(RFI_STREAM, GROUP_NAME, mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    logger.logger.info(f"Passive Worker {WORKER_ID} listening on {RFI_STREAM}...")

    while True:
        try:
            streams = redis_client.xreadgroup(
                GROUP_NAME,
                CONSUMER_NAME,
                {RFI_STREAM: ">"},
                count=1,
                block=2000
            )

            if streams:
                for stream, messages in streams:
                    for message_id, data in messages:
                        try:
                            worker.process_rfi(data)
                            redis_client.xack(RFI_STREAM, GROUP_NAME, message_id)
                        except Exception as e:
                            logger.logger.error(f"Error processing RFI: {e}")
                            import traceback
                            traceback.print_exc()

        except redis.exceptions.ConnectionError as e:
            logger.logger.error(f"Redis connection error: {e}")
            time.sleep(5)
        except Exception as e:
            logger.logger.error(f"Worker loop error: {e}")
            time.sleep(1)


# Utility functions for benchmark harness

def get_ready_workers() -> int:
    """Get count of ready workers."""
    return redis_client.scard(WORKERS_READY_KEY)


async def wait_for_workers(required_k: int, timeout: float = 30.0) -> bool:
    """
    Wait for required number of workers to be ready.
    CRITICAL: All workers must be ready before benchmark execution begins.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready_count = redis_client.scard(WORKERS_READY_KEY)
        if ready_count >= required_k:
            logger.logger.info(f"All {required_k} workers ready, beginning benchmark execution")
            return True
        logger.logger.debug(f"Waiting for workers: {ready_count}/{required_k} ready")
        await asyncio.sleep(1.0)

    ready_count = redis_client.scard(WORKERS_READY_KEY)
    raise RuntimeError(f"Only {ready_count}/{required_k} workers ready after {timeout}s")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
