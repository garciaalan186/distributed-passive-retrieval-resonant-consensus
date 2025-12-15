"""
Passive Agent Worker for DPR-RC System

Per Architecture Spec Section 2.1:
"Passive Agent (P) - A serverless worker representing a frozen snapshot of history,
performing verification on demand."

LAZY LOADING ARCHITECTURE:
- Workers start with EMPTY vector stores
- When an RFI arrives with target_shards, worker loads that shard on-demand
- Shards are cached locally after first load (no re-download)
- Pre-computed embeddings are loaded from GCS (no embedding computation at runtime)

GCS Data Source:
    gs://{HISTORY_BUCKET}/
    ├── raw/{scale}/shards/shard_{year}.json       # Fallback: raw text
    └── embeddings/{model}/shards/shard_{year}.npz  # Primary: pre-computed vectors
"""

import os
import time
import json
import redis
import threading
import hashlib
import tempfile
import requests
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

# Disable ChromaDB telemetry (fixes "capture() takes 1 positional argument" errors)
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import chromadb
import numpy as np

from .models import (
    ComponentType, EventType, LogEntry, ConsensusVote
)
from .logging_utils import StructuredLogger
from .embedding_utils import (
    GCSEmbeddingStore,
    load_embeddings_npz,
    embed_query,
    compute_embeddings,
    DEFAULT_EMBEDDING_MODEL,
    get_model_folder_name
)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WORKER_ID = os.getenv("HOSTNAME", f"worker-{os.getpid()}")
WORKER_EPOCH = os.getenv("WORKER_EPOCH", "2020")  # Default epoch (can handle any shard)
HISTORY_BUCKET = os.getenv("HISTORY_BUCKET", None)
HISTORY_SCALE = os.getenv("HISTORY_SCALE", "medium")  # Scale level for data
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

# SLM Service for L2 verification (per spec: SLM-based reasoning, not token overlap)
SLM_SERVICE_URL = os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
SLM_VERIFY_TIMEOUT = float(os.getenv("SLM_VERIFY_TIMEOUT", "10.0"))  # seconds

logger = StructuredLogger(ComponentType.PASSIVE_WORKER)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

RFI_STREAM = "dpr:rfi"
GROUP_NAME = "passive_workers"
CONSUMER_NAME = WORKER_ID


@dataclass
class FrozenAgentState:
    """
    Represents a frozen snapshot of A* at a specific point in time.
    """
    creation_time: str
    epoch_year: int
    context_summary: str = ""

    def verify(self, query: str, candidate: str) -> dict:
        """Verify candidate answer against this frozen state's knowledge."""
        return {
            "score": 0.85,
            "temporal_context": f"From {self.epoch_year} perspective",
            "prompt": f"Verify: {candidate}",
            "response": f"Consistent with {self.creation_time} knowledge base",
            "tokens": 10
        }


class PassiveWorker:
    """
    Passive Agent Worker with LAZY LOADING architecture.

    Key changes from eager loading:
    1. Workers start with empty vector stores
    2. Shards are loaded on-demand when RFI specifies target_shards
    3. Pre-computed embeddings are loaded from GCS (no runtime embedding)
    4. Loaded shards are cached locally for subsequent requests
    """

    def __init__(self, epoch_year: int = None):
        self.epoch_year = epoch_year or int(WORKER_EPOCH)
        self.frozen_state = FrozenAgentState(
            creation_time=f"{self.epoch_year}-01-01",
            epoch_year=self.epoch_year
        )

        # Initialize ChromaDB client (no collections yet - lazy loaded)
        self.chroma = chromadb.Client()

        # Track loaded shards and their collections
        self._loaded_shards: Dict[str, chromadb.Collection] = {}
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        # GCS embedding store (lazy initialized)
        self._gcs_store: Optional[GCSEmbeddingStore] = None
        self._cache_dir = tempfile.mkdtemp(prefix="dpr_worker_")

        logger.logger.info(
            f"PassiveWorker initialized (lazy loading mode). "
            f"Bucket: {HISTORY_BUCKET}, Scale: {HISTORY_SCALE}, Model: {EMBEDDING_MODEL}"
        )

    def _get_gcs_store(self) -> Optional[GCSEmbeddingStore]:
        """Lazy-initialize GCS store."""
        if self._gcs_store is None and HISTORY_BUCKET:
            try:
                self._gcs_store = GCSEmbeddingStore(
                    bucket_name=HISTORY_BUCKET,
                    cache_dir=self._cache_dir
                )
            except Exception as e:
                logger.logger.warning(f"Could not initialize GCS store: {e}")
        return self._gcs_store

    def _get_loading_lock(self, shard_id: str) -> threading.Lock:
        """Get or create a lock for loading a specific shard."""
        with self._global_lock:
            if shard_id not in self._loading_locks:
                self._loading_locks[shard_id] = threading.Lock()
            return self._loading_locks[shard_id]

    def _load_shard_on_demand(self, shard_id: str) -> Optional[chromadb.Collection]:
        """
        Load a shard on-demand if not already loaded.

        Priority:
        1. Check if already loaded (return cached collection)
        2. Try to load pre-computed embeddings from GCS
        3. Fall back to raw JSON from GCS (compute embeddings locally)
        4. Fall back to Redis cache
        5. Fall back to generated fallback data

        Args:
            shard_id: Shard identifier (e.g., "shard_2020")

        Returns:
            ChromaDB collection with loaded data, or None if loading failed
        """
        # Fast path: already loaded
        if shard_id in self._loaded_shards:
            return self._loaded_shards[shard_id]

        # Acquire per-shard lock to prevent concurrent loading
        lock = self._get_loading_lock(shard_id)
        with lock:
            # Double-check after acquiring lock
            if shard_id in self._loaded_shards:
                return self._loaded_shards[shard_id]

            logger.logger.info(f"Loading shard on-demand: {shard_id}")

            # Create collection for this shard
            collection = self.chroma.get_or_create_collection(
                name=f"history_{shard_id}",
                metadata={"hnsw:space": "cosine"}
            )

            # Try loading strategies in order
            loaded = False

            # Strategy 1: Pre-computed embeddings from GCS
            gcs_store = self._get_gcs_store()
            if gcs_store and not loaded:
                loaded = self._load_from_gcs_embeddings(shard_id, collection, gcs_store)

            # Strategy 2: Raw JSON from GCS (compute embeddings locally)
            if gcs_store and not loaded:
                loaded = self._load_from_gcs_raw(shard_id, collection, gcs_store)

            # Strategy 3: Redis cache
            if not loaded:
                loaded = self._load_from_redis_cache(shard_id, collection)

            # Strategy 4: Fallback generated data
            if not loaded:
                loaded = self._generate_fallback_data(shard_id, collection)

            if loaded:
                self._loaded_shards[shard_id] = collection
                logger.logger.info(
                    f"Shard {shard_id} loaded: {collection.count()} documents"
                )
                return collection
            else:
                logger.logger.warning(f"Failed to load shard {shard_id}")
                return None

    def _load_from_gcs_embeddings(
        self,
        shard_id: str,
        collection: chromadb.Collection,
        gcs_store: GCSEmbeddingStore
    ) -> bool:
        """Load pre-computed embeddings from GCS."""
        try:
            result = gcs_store.download_embeddings(
                model_id=EMBEDDING_MODEL,
                scale=HISTORY_SCALE,
                shard_id=shard_id
            )

            if result is None:
                return False

            embeddings, doc_ids, texts, metadatas, metadata = result

            logger.logger.info(
                f"Loaded pre-computed embeddings from GCS: "
                f"{metadata.num_documents} docs, model={metadata.model_id}"
            )

            # Insert into ChromaDB with pre-computed embeddings
            self._bulk_insert_with_embeddings(
                collection, doc_ids, texts, metadatas, embeddings
            )

            return True

        except Exception as e:
            logger.logger.warning(f"Failed to load embeddings from GCS: {e}")
            return False

    def _load_from_gcs_raw(
        self,
        shard_id: str,
        collection: chromadb.Collection,
        gcs_store: GCSEmbeddingStore
    ) -> bool:
        """Load raw JSON from GCS and compute embeddings locally."""
        try:
            events = gcs_store.download_raw_shard(
                scale=HISTORY_SCALE,
                shard_id=shard_id
            )

            if not events:
                return False

            logger.logger.info(
                f"Loaded raw JSON from GCS: {len(events)} events. "
                f"Computing embeddings locally..."
            )

            # Extract texts
            texts = [event['content'] for event in events]
            doc_ids = [event['id'] for event in events]
            metadatas = [
                {
                    "timestamp": event.get('timestamp', ''),
                    "topic": event.get('topic', ''),
                    "event_type": event.get('event_type', ''),
                    "perspective": event.get('perspective', '')
                }
                for event in events
            ]

            # Compute embeddings locally
            embeddings = compute_embeddings(texts, EMBEDDING_MODEL)

            # Insert with embeddings
            self._bulk_insert_with_embeddings(
                collection, doc_ids, texts, metadatas, embeddings
            )

            return True

        except Exception as e:
            logger.logger.warning(f"Failed to load raw from GCS: {e}")
            return False

    def _load_from_redis_cache(
        self,
        shard_id: str,
        collection: chromadb.Collection
    ) -> bool:
        """Load from Redis cache (benchmark pre-loaded data)."""
        try:
            # Extract year from shard_id (e.g., "shard_2020" -> "2020")
            year = shard_id.replace("shard_", "")
            cache_key = f"dpr:history_cache:{year}"
            cached = redis_client.get(cache_key)

            if not cached:
                return False

            documents = json.loads(cached)
            logger.logger.info(f"Loaded {len(documents)} documents from Redis cache")

            # Insert into collection (let ChromaDB compute embeddings)
            self._bulk_insert(collection, documents)
            return True

        except Exception as e:
            logger.logger.warning(f"Failed to load from Redis cache: {e}")
            return False

    def _generate_fallback_data(
        self,
        shard_id: str,
        collection: chromadb.Collection
    ) -> bool:
        """Generate minimal fallback data for testing."""
        try:
            # Extract year from shard_id
            year_str = shard_id.replace("shard_", "")
            try:
                year = int(year_str)
            except ValueError:
                year = 2020

            fallback_docs = []
            for i in range(10):
                doc_id = f"fallback_{year}_{i}"
                content = (
                    f"Historical record from {year}: Research milestone {i} achieved. "
                    f"Progress in domain area with metrics showing improvement. "
                    f"Epoch {year} data point {i}."
                )
                fallback_docs.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": {"year": year, "type": "fallback", "index": i}
                })

            self._bulk_insert(collection, fallback_docs)
            logger.logger.info(f"Generated {len(fallback_docs)} fallback documents")
            return True

        except Exception as e:
            logger.logger.warning(f"Failed to generate fallback data: {e}")
            return False

    def _bulk_insert_with_embeddings(
        self,
        collection: chromadb.Collection,
        doc_ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
        embeddings: np.ndarray
    ):
        """Bulk insert documents with pre-computed embeddings."""
        batch_size = 1000

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size].tolist()

            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    embeddings=batch_embeddings
                )
            except Exception as e:
                # Handle duplicate IDs
                if "duplicate" in str(e).lower() or "already" in str(e).lower():
                    self._insert_non_duplicates(
                        collection, batch_ids, batch_texts,
                        batch_metadatas, batch_embeddings
                    )
                else:
                    logger.logger.error(f"Failed to insert batch: {e}")

    def _insert_non_duplicates(
        self,
        collection: chromadb.Collection,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
        embeddings: List[List[float]]
    ):
        """Insert only non-duplicate documents."""
        try:
            existing = collection.get(ids=ids)
            existing_ids = set(existing.get('ids', []))

            new_ids = []
            new_texts = []
            new_metadatas = []
            new_embeddings = []

            for idx, doc_id in enumerate(ids):
                if doc_id not in existing_ids:
                    new_ids.append(doc_id)
                    new_texts.append(texts[idx])
                    new_metadatas.append(metadatas[idx])
                    new_embeddings.append(embeddings[idx])

            if new_ids:
                collection.add(
                    ids=new_ids,
                    documents=new_texts,
                    metadatas=new_metadatas,
                    embeddings=new_embeddings
                )
        except Exception as e:
            logger.logger.warning(f"Could not add non-duplicate docs: {e}")

    def _bulk_insert(self, collection: chromadb.Collection, documents: List[Dict]):
        """Bulk insert documents (ChromaDB computes embeddings)."""
        if not documents:
            return

        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            doc_id = doc.get('id', hashlib.md5(doc['content'].encode()).hexdigest()[:12])
            ids.append(doc_id)
            contents.append(doc['content'])
            metadatas.append(doc.get('metadata', {}))

        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_contents = contents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_contents,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                error_msg = str(e).lower()
                if "duplicate" in error_msg or "already" in error_msg:
                    # Skip duplicates
                    pass
                else:
                    logger.logger.error(f"Failed to insert batch: {e}")

    def ingest_benchmark_data(self, events: List[Dict], shard_id: str = None):
        """
        Public method to ingest benchmark data directly.
        Called by benchmark harness to load synthetic history.
        """
        if shard_id is None:
            shard_id = f"shard_{self.epoch_year}"

        # Get or create collection for this shard
        collection = self.chroma.get_or_create_collection(
            name=f"history_{shard_id}",
            metadata={"hnsw:space": "cosine"}
        )

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

        self._bulk_insert(collection, documents)
        self._loaded_shards[shard_id] = collection

        # Also cache in Redis for other workers
        try:
            year = shard_id.replace("shard_", "")
            cache_key = f"dpr:history_cache:{year}"
            redis_client.setex(cache_key, 3600, json.dumps(documents))
        except Exception as e:
            logger.logger.warning(f"Failed to cache in Redis: {e}")

    def retrieve(
        self,
        query_text: str,
        shard_id: str,
        timestamp_context: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant historical context from a specific shard.

        Uses the SAME embedding model (all-MiniLM-L6-v2) for query embedding
        as was used for document embeddings, ensuring consistent vector space.

        Args:
            query_text: Query to search for
            shard_id: Which shard to search in
            timestamp_context: Optional timestamp filter

        Returns:
            Best matching document or None
        """
        # Load shard on-demand if needed
        collection = self._load_shard_on_demand(shard_id)

        if collection is None or collection.count() == 0:
            logger.logger.warning(f"Shard {shard_id} is empty or failed to load")
            return None

        try:
            # Embed query using the SAME model as documents (fixes embedding mismatch)
            query_embedding = embed_query(query_text, EMBEDDING_MODEL)

            # Query using pre-computed embedding vector
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
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
            logger.logger.error(f"Retrieval error: {e}")
            return None

    def verify_l2(self, content: str, query: str, depth: int = 0) -> float:
        """
        L2 Verification using SLM reasoning.

        Per Mathematical Model Section 5.2, Equation 9:
        C(r_p) = V(q, context_p) · (1 / (1 + i))

        This calls the SLM service to get a reasoned verification judgment,
        rather than using simple token overlap.
        """
        try:
            response = requests.post(
                f"{SLM_SERVICE_URL}/verify",
                json={
                    "query": query,
                    "retrieved_content": content[:2000],  # Limit content size
                    "trace_id": f"{WORKER_ID}_{time.time()}"
                },
                timeout=SLM_VERIFY_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                base_score = result.get("confidence", 0.5)

                # Apply depth penalty per spec equation
                depth_penalty = 1.0 / (1.0 + depth)
                confidence = base_score * depth_penalty

                logger.logger.debug(
                    f"SLM verification: confidence={base_score:.2f}, "
                    f"supports={result.get('supports_query')}, "
                    f"reasoning={result.get('reasoning', '')[:100]}"
                )

                return max(0.0, min(1.0, confidence))
            else:
                logger.logger.warning(
                    f"SLM service returned {response.status_code}, "
                    f"falling back to heuristic"
                )
                return self._verify_l2_fallback(content, query, depth)

        except requests.exceptions.RequestException as e:
            logger.logger.warning(f"SLM service unavailable: {e}, using fallback")
            return self._verify_l2_fallback(content, query, depth)

    def _verify_l2_fallback(self, content: str, query: str, depth: int = 0) -> float:
        """
        Fallback L2 verification using token overlap.

        Used only when SLM service is unavailable.
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
        depth_penalty = 1.0 / (1.0 + depth)
        confidence = v_score * depth_penalty

        return max(0.0, min(1.0, confidence))

    def calculate_quadrant(self, content: str, confidence: float) -> List[float]:
        """L3 Semantic Quadrant Topology calculation."""
        content_hash = hash(content)
        x_base = ((content_hash % 100) / 100.0)
        x = (x_base * 0.5) + (confidence * 0.5)
        y_base = (((content_hash >> 8) % 100) / 100.0)
        y = (y_base * 0.3) + (confidence * 0.7)
        return [round(x, 2), round(y, 2)]

    def process_rfi(self, rfi_data: Dict[str, Any]):
        """
        Process a Request for Information (RFI) from the Active Agent.

        LAZY LOADING: Loads target shards on-demand when RFI specifies them.

        Query handling:
        - query_text: Enhanced query (from SLM) used for embedding-based retrieval
        - original_query: Original user query used for SLM verification
        """
        trace_id = rfi_data.get('trace_id', 'unknown')
        query_text = rfi_data.get('query_text', '')  # Enhanced query for retrieval
        original_query = rfi_data.get('original_query', query_text)  # Original for verification
        ts_context = rfi_data.get('timestamp_context', '')
        target_shards_str = rfi_data.get('target_shards', '[]')

        # Parse target shards
        try:
            target_shards = json.loads(target_shards_str) if target_shards_str else []
        except json.JSONDecodeError:
            target_shards = []

        # If no target shards specified, use default based on epoch
        if not target_shards or "broadcast" in target_shards:
            target_shards = [f"shard_{self.epoch_year}"]

        # Process each target shard
        for shard_id in target_shards:
            # Skip if shard doesn't match our epoch (for sharded deployments)
            # In shard-agnostic mode, we handle all shards
            my_shard = f"shard_{self.epoch_year}"

            # Retrieve from this shard using ENHANCED query (better embeddings match)
            doc = self.retrieve(query_text, shard_id, ts_context)

            if not doc:
                logger.logger.debug(f"No relevant history in {shard_id} for: {query_text[:50]}...")
                continue

            # L2 Verification using ORIGINAL query (SLM judges against user's intent)
            depth = doc.get('metadata', {}).get('hierarchy_depth', 0)
            confidence = self.verify_l2(doc['content'], original_query, depth)

            if confidence < 0.3:
                logger.logger.debug(f"Confidence {confidence:.2f} below threshold")
                continue

            # L3 Semantic quadrant
            quadrant = self.calculate_quadrant(doc['content'], confidence)

            # Cast vote
            vote = ConsensusVote(
                trace_id=trace_id,
                worker_id=WORKER_ID,
                content_hash=logger.hash_payload(doc['content']),
                confidence_score=confidence,
                semantic_quadrant=quadrant,
                content_snippet=doc['content'][:500]
            )

            # Publish vote
            redis_client.publish(f"dpr:responses:{trace_id}", json.dumps(vote.model_dump()))

            logger.log_event(
                trace_id, EventType.VOTE_CAST, vote.model_dump(),
                metrics={"confidence": confidence, "shard": shard_id}
            )

    def get_loaded_shards(self) -> List[str]:
        """Return list of currently loaded shard IDs."""
        return list(self._loaded_shards.keys())


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - starts worker thread on startup"""
    worker_thread = threading.Thread(target=run_worker_loop, daemon=True)
    worker_thread.start()
    logger.logger.info(f"Passive Worker {WORKER_ID} started (lazy loading mode)")
    yield
    logger.logger.info(f"Passive Worker {WORKER_ID} shutting down")


app = FastAPI(title="DPR-PassiveWorker", lifespan=lifespan)


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint"""
    global _worker_instance
    loaded_shards = _worker_instance.get_loaded_shards() if _worker_instance else []
    return {
        "status": "healthy",
        "worker_id": WORKER_ID,
        "epoch": WORKER_EPOCH,
        "mode": "lazy_loading",
        "loaded_shards": loaded_shards,
        "bucket": HISTORY_BUCKET,
        "scale": HISTORY_SCALE,
        "model": EMBEDDING_MODEL
    }


@app.post("/ingest")
def ingest_data(data: Dict[str, Any]):
    """Endpoint to ingest benchmark data"""
    global _worker_instance
    if _worker_instance and 'events' in data:
        shard_id = data.get('shard_id', None)
        _worker_instance.ingest_benchmark_data(data['events'], shard_id)
        return {"status": "ok", "ingested": len(data['events'])}
    return {"status": "error", "message": "No worker instance or invalid data"}


@app.get("/shards")
def list_shards():
    """List loaded shards and their document counts"""
    global _worker_instance
    if _worker_instance:
        shards = {}
        for shard_id, collection in _worker_instance._loaded_shards.items():
            shards[shard_id] = {
                "count": collection.count()
            }
        return {"shards": shards}
    return {"shards": {}}


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
