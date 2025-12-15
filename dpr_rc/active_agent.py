import asyncio
import json
import os
import time
import math
import requests
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
import uuid

from .models import (
    QueryRequest, LogEntry, ComponentType, EventType,
    ConsensusVote, RetrievalResult
)
from .logging_utils import StructuredLogger
from .debug_utils import (
    debug_query_received, debug_query_enhancement, debug_routing,
    debug_http_worker_call, debug_http_worker_response,
    debug_consensus_calculation, debug_final_response,
    debug_log, DEBUG_BREAKPOINTS
)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
RESPONSE_TIMEOUT = float(os.getenv("RESPONSE_TIMEOUT", "5.0"))  # seconds to wait for votes
MIN_VOTES_FOR_CONSENSUS = int(os.getenv("MIN_VOTES", "1"))  # Minimum votes needed

# SLM Service for query enhancement (improves retrieval quality)
SLM_SERVICE_URL = os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
SLM_ENHANCE_TIMEOUT = float(os.getenv("SLM_ENHANCE_TIMEOUT", "5.0"))  # seconds
ENABLE_QUERY_ENHANCEMENT = os.getenv("ENABLE_QUERY_ENHANCEMENT", "true").lower() == "true"

# HTTP-based worker communication (for Cloud Run without Redis)
# Can be comma-separated list of worker URLs
PASSIVE_WORKER_URL = os.getenv("PASSIVE_WORKER_URL", "")
USE_HTTP_WORKERS = os.getenv("USE_HTTP_WORKERS", "true").lower() == "true"
HTTP_WORKER_TIMEOUT = float(os.getenv("HTTP_WORKER_TIMEOUT", "30.0"))  # seconds

logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

# Initialize Redis client (may fail in HTTP-only mode, that's OK)
try:
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()  # Test connection
    REDIS_AVAILABLE = True
except Exception as e:
    logger.logger.warning(f"Redis not available: {e}. Using HTTP-only mode.")
    redis_client = None
    REDIS_AVAILABLE = False

# Streams
RFI_STREAM = "dpr:rfi"
VOTE_STREAM = "dpr:votes"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    # Startup
    if REDIS_AVAILABLE and redis_client:
        try:
            redis_client.xgroup_create(VOTE_STREAM, "controller_group", mkstream=True)
        except Exception:
            pass  # Group already exists
    logger.logger.info(
        f"Active Controller Started (Redis: {REDIS_AVAILABLE}, HTTP Workers: {USE_HTTP_WORKERS})"
    )
    yield
    # Shutdown
    logger.logger.info("Active Controller Shutting Down")


app = FastAPI(title="DPR-ActiveController", lifespan=lifespan)


class RouteLogic:
    """L1 Time-Sharded Routing Logic per DPR Architecture Spec Section 3.1.1"""

    @staticmethod
    def get_target_shards(query: QueryRequest) -> List[str]:
        """
        Determine target shards based on timestamp context.
        Per spec: "The Central Index is partitioned into Time-Based Shards"
        """
        if query.timestamp_context:
            # Deterministic sharding logic based on year
            year = query.timestamp_context[:4]
            return [f"shard_{year}"]
        return ["broadcast"]


# Per spec Section 2.1: "Passive Agent (P) - A serverless worker representing
# a frozen snapshot of history"
@dataclass
class FrozenAgentState:
    """
    Represents a frozen snapshot of A* at a specific point in time.
    Per Mathematical Model Section 4.2: "Each historical agent A_k represents
    a frozen snapshot of the context state from a previous time step."
    """
    creation_time: str

    def verify(self, query: str, candidate: str) -> dict:
        """
        Simulate temporal understanding verification.
        In production, this runs inference on the frozen model checkpoint.
        """
        return {
            "score": 0.9,
            "prompt": f"Verify: {candidate}",
            "response": f"Consistent with {self.creation_time} knowledge",
            "tokens": 10
        }


@dataclass
class QuadrantClassification:
    """Classification result for Semantic Quadrant Topology"""
    name: str
    reasoning: str


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "component": "active_controller",
        "query_enhancement": ENABLE_QUERY_ENHANCEMENT,
        "slm_service": SLM_SERVICE_URL,
        "redis_available": REDIS_AVAILABLE,
        "http_workers": USE_HTTP_WORKERS,
        "passive_worker_url": PASSIVE_WORKER_URL
    }


def call_workers_via_http(
    trace_id: str,
    query_text: str,
    original_query: str,
    target_shards: List[str],
    timestamp_context: Optional[str] = None
) -> List[ConsensusVote]:
    """
    Call passive workers directly via HTTP instead of Redis.

    This enables Cloud Run deployments without Redis/VPC.

    Args:
        trace_id: Trace ID for logging
        query_text: Enhanced query for retrieval
        original_query: Original query for verification
        target_shards: List of shard IDs to query
        timestamp_context: Optional timestamp context

    Returns:
        List of ConsensusVote objects from workers
    """
    if not PASSIVE_WORKER_URL:
        logger.logger.warning("PASSIVE_WORKER_URL not set, cannot call workers via HTTP")
        return []

    # Support multiple worker URLs (comma-separated)
    worker_urls = [url.strip() for url in PASSIVE_WORKER_URL.split(",") if url.strip()]

    all_votes = []

    for worker_url in worker_urls:
        try:
            # Ensure URL has /process_rfi endpoint
            endpoint = worker_url.rstrip("/")
            if not endpoint.endswith("/process_rfi"):
                endpoint = f"{endpoint}/process_rfi"

            request_payload = {
                "trace_id": trace_id,
                "query_text": query_text,
                "original_query": original_query,
                "target_shards": target_shards,
                "timestamp_context": timestamp_context
            }

            # DEBUG: HTTP worker call
            debug_http_worker_call(trace_id, endpoint, request_payload)

            response = requests.post(
                endpoint,
                json=request_payload,
                timeout=HTTP_WORKER_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                votes_data = result.get("votes", [])
                worker_id = result.get("worker_id", "unknown")

                # DEBUG: HTTP worker response
                debug_http_worker_response(
                    trace_id, endpoint, len(votes_data),
                    {"worker_id": worker_id, "votes": votes_data, "shards": result.get("shards_queried", [])}
                )

                for vote_data in votes_data:
                    try:
                        vote = ConsensusVote(**vote_data)
                        all_votes.append(vote)
                        logger.logger.debug(
                            f"Received vote from {worker_id}: confidence={vote.confidence_score:.2f}"
                        )
                    except Exception as e:
                        logger.logger.warning(f"Failed to parse vote: {e}")
            else:
                # DEBUG: Worker error
                debug_log("ActiveController", f"Worker error: {response.status_code}",
                         {"url": worker_url, "response": response.text[:500]})
                logger.logger.warning(
                    f"Worker {worker_url} returned {response.status_code}: {response.text[:200]}"
                )

        except requests.exceptions.RequestException as e:
            logger.logger.warning(f"Failed to call worker {worker_url}: {e}")

    logger.logger.info(f"HTTP workers returned {len(all_votes)} votes from {len(worker_urls)} workers")
    return all_votes


def enhance_query_via_slm(query_text: str, timestamp_context: Optional[str] = None) -> dict:
    """
    Call the SLM service to enhance the query for better retrieval.

    The SLM:
    - Expands abbreviations (ML -> machine learning)
    - Adds synonyms for better recall
    - Clarifies ambiguous terms
    - Incorporates temporal context

    Args:
        query_text: Original query from user
        timestamp_context: Optional temporal context

    Returns:
        dict with enhanced_query and expansions, or original query on failure
    """
    if not ENABLE_QUERY_ENHANCEMENT:
        return {
            "original_query": query_text,
            "enhanced_query": query_text,
            "expansions": [],
            "enhancement_used": False
        }

    try:
        response = requests.post(
            f"{SLM_SERVICE_URL}/enhance_query",
            json={
                "query": query_text,
                "timestamp_context": timestamp_context
            },
            timeout=SLM_ENHANCE_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            logger.logger.info(
                f"Query enhanced: '{query_text}' -> '{result.get('enhanced_query', query_text)}' "
                f"(expansions: {result.get('expansions', [])})"
            )
            return {
                "original_query": query_text,
                "enhanced_query": result.get("enhanced_query", query_text),
                "expansions": result.get("expansions", []),
                "enhancement_used": True,
                "inference_time_ms": result.get("inference_time_ms", 0)
            }
        else:
            logger.logger.warning(
                f"SLM enhance_query returned {response.status_code}, using original query"
            )

    except requests.exceptions.RequestException as e:
        logger.logger.warning(f"SLM service unavailable for query enhancement: {e}")

    # Fallback: return original query
    return {
        "original_query": query_text,
        "enhanced_query": query_text,
        "expansions": [],
        "enhancement_used": False
    }


@app.post("/query", response_model=RetrievalResult)
async def handle_query(request: QueryRequest):
    """
    Main query handler implementing the DPR Protocol (Spec Section 4).

    Flow:
    1. Gap Detection & Routing (L1)
    2. SUBSCRIBE to response channel BEFORE broadcast (CRITICAL - fixes race condition)
    3. Targeted RFI Broadcast
    4. Gather Votes (L2 Verification results)
    5. Resonant Consensus (L3)
    6. Superposition Injection
    """
    trace_id = request.trace_id
    logger.log_event(trace_id, EventType.SYSTEM_INIT, request.model_dump())

    # DEBUG: Query received
    debug_query_received(trace_id, request.query_text, request.timestamp_context)

    pubsub = None
    try:
        # 0. Query Enhancement via SLM (improves retrieval quality)
        # The SLM expands abbreviations, adds synonyms, and clarifies ambiguous terms
        enhancement_result = enhance_query_via_slm(
            request.query_text,
            request.timestamp_context
        )
        enhanced_query = enhancement_result["enhanced_query"]

        # DEBUG: Query enhancement result
        debug_query_enhancement(
            trace_id,
            original=request.query_text,
            enhanced=enhanced_query,
            expansions=enhancement_result.get("expansions", []),
            used=enhancement_result.get("enhancement_used", False),
            time_ms=enhancement_result.get("inference_time_ms", 0)
        )

        logger.log_event(trace_id, EventType.SYSTEM_INIT, {
            "original_query": request.query_text,
            "enhanced_query": enhanced_query,
            "expansions": enhancement_result.get("expansions", []),
            "enhancement_used": enhancement_result.get("enhancement_used", False)
        })

        # 1. L1 Routing - Determine target shards
        target_shards = RouteLogic.get_target_shards(request)

        # DEBUG: Routing decision
        debug_routing(trace_id, enhanced_query, target_shards)

        # 2. Gather Votes - Use HTTP workers or Redis depending on availability
        votes: List[ConsensusVote] = []

        # Try HTTP workers first (preferred for Cloud Run)
        if USE_HTTP_WORKERS and PASSIVE_WORKER_URL:
            logger.logger.info(f"Calling workers via HTTP: {PASSIVE_WORKER_URL}")
            votes = call_workers_via_http(
                trace_id=trace_id,
                query_text=enhanced_query,
                original_query=request.query_text,
                target_shards=target_shards,
                timestamp_context=request.timestamp_context
            )
            logger.log_event(trace_id, EventType.RFI_BROADCAST, {
                "method": "http",
                "target_shards": target_shards,
                "votes_received": len(votes)
            })

        # Fall back to Redis if HTTP didn't get votes and Redis is available
        if not votes and REDIS_AVAILABLE and redis_client:
            logger.logger.info("Falling back to Redis for vote collection")

            # Subscribe to response channel BEFORE broadcasting RFI
            pubsub = redis_client.pubsub()
            response_channel = f"dpr:responses:{trace_id}"
            pubsub.subscribe(response_channel)

            # Small delay to ensure subscription is fully active
            await asyncio.sleep(0.05)

            # Broadcast RFI to Redis Stream
            rfi_payload = {
                "trace_id": trace_id,
                "query_text": enhanced_query,
                "original_query": request.query_text,
                "target_shards": json.dumps(target_shards),
                "timestamp_context": request.timestamp_context or ""
            }
            redis_client.xadd(RFI_STREAM, rfi_payload)
            logger.log_event(trace_id, EventType.RFI_BROADCAST, {
                "method": "redis",
                **rfi_payload
            })

            # Wait for votes via Redis Pub/Sub
            start_time = time.time()
            while (time.time() - start_time) < RESPONSE_TIMEOUT:
                message = pubsub.get_message(ignore_subscribe_messages=True)
                if message and message.get('data'):
                    try:
                        data = message['data']
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        vote_data = json.loads(data)
                        votes.append(ConsensusVote(**vote_data))

                        if len(votes) >= MIN_VOTES_FOR_CONSENSUS:
                            await asyncio.sleep(0.3)
                            while True:
                                msg = pubsub.get_message(ignore_subscribe_messages=True)
                                if not msg or not msg.get('data'):
                                    break
                                try:
                                    d = msg['data']
                                    if isinstance(d, bytes):
                                        d = d.decode('utf-8')
                                    votes.append(ConsensusVote(**json.loads(d)))
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            break
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.logger.warning(f"Failed to parse vote: {e}")
                await asyncio.sleep(0.05)

            # Cleanup subscription
            pubsub.unsubscribe(response_channel)
            pubsub.close()
            pubsub = None

        # Handle no votes case
        if not votes:
            logger.log_event(trace_id, EventType.HALLUCINATION_DETECTED,
                           {"reason": "No votes received"})
            # DEBUG: No votes received
            debug_final_response(
                trace_id, status="FAILED", confidence=0.0,
                answer="No consensus reached.", sources=[]
            )
            return RetrievalResult(
                trace_id=trace_id,
                final_answer="No consensus reached.",
                confidence=0.0,
                status="FAILED",
                sources=[]
            )

        # 5. Resonant Consensus Protocol (L3) - Per Spec Section 4.4
        # "Instead of returning the single best answer, the system enters a Consensus Phase"

        # Group votes by content hash (same content = same claim)
        unique_candidates: Dict[str, Dict] = {}
        for v in votes:
            if v.content_hash not in unique_candidates:
                unique_candidates[v.content_hash] = {
                    "content": v.content_snippet,
                    "votes": []
                }
            unique_candidates[v.content_hash]["votes"].append(v)

        # Classify into Semantic Quadrants per Mathematical Model Section 6.2
        # Symmetric Resonance (Consensus): v+ > 0 ∧ v- > 0 - high agreement
        # Asymmetry (Perspective): partial truth valid from specific perspectives
        consensus_set = []
        perspectival_set = []

        for chash, data in unique_candidates.items():
            scores = [v.confidence_score for v in data["votes"]]
            vote_count = len(scores)

            if not scores:
                continue

            mean_score = sum(scores) / len(scores)
            variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
            std_score = math.sqrt(variance)

            # Quadrant Classification per spec
            # High mean + low std = strong consensus (Symmetric Resonance)
            # Per Mathematical Model: "High-entropy bridge concepts agreed upon by diverse contexts"
            if mean_score > 0.7 and std_score < 0.2:
                quadrant = "SYMMETRIC_RESONANCE"
                consensus_set.append(data["content"])
            elif mean_score > 0.4:
                quadrant = "ASYMMETRIC"
                perspectival_set.append({
                    "claim": data["content"],
                    "snapshot_views": {v.worker_id: v.confidence_score for v in data["votes"]},
                    "quadrant": quadrant,
                    "metrics": {"mean": mean_score, "std": std_score, "vote_count": vote_count}
                })
            else:
                quadrant = "DISSONANT_POLARIZATION"
                # Include significantly dissonant claims as perspectives if they have some support
                if mean_score > 0.3:
                    perspectival_set.append({
                        "claim": data["content"],
                        "snapshot_views": {v.worker_id: v.confidence_score for v in data["votes"]},
                        "quadrant": quadrant,
                        "metrics": {"mean": mean_score, "std": std_score, "vote_count": vote_count}
                    })

        # DEBUG: Consensus calculation results
        debug_consensus_calculation(
            trace_id,
            votes_count=len(votes),
            unique_candidates=len(unique_candidates),
            consensus_set=consensus_set,
            perspectival_set=perspectival_set
        )

        # 6. Superposition Injection - Per Spec Section 4.5
        # "A* injects a Superposition Object into its context, containing both
        # the Consensus Truth and distinct Perspectives"
        superposition_object = {
            "consensus_facts": consensus_set,
            "perspectival_claims": perspectival_set
        }

        # Generate Response (A*) - Construct answer from superposition
        if consensus_set:
            final_answer = " ".join(consensus_set)
            if perspectival_set:
                final_answer += "\n\nAdditionally, there are evolving perspectives: " + \
                               "; ".join([p['claim'] for p in perspectival_set])
            confidence = 0.95
        elif perspectival_set:
            # Uncertainty case: Present options per spec
            # "allows A* to generate a nuanced reply acknowledging both the agreed facts
            # and the conflicting perspectives"
            options = [f"- {p['claim']} (Agreement: {p['metrics']['mean']:.2f})"
                      for p in perspectival_set]
            final_answer = "The historical record shows varying perspectives:\n" + "\n".join(options)
            confidence = 0.7
        else:
            final_answer = "No relevant information found in the historical record."
            confidence = 0.0

        logger.log_event(trace_id, EventType.CONSENSUS_REACHED, {
            "superposition": superposition_object,
            "final_answer": final_answer[:200],
            "num_votes": len(votes),
            "num_consensus": len(consensus_set),
            "num_perspectival": len(perspectival_set)
        })

        # DEBUG: Final response
        status = "SUCCESS" if confidence > 0 else "NO_DATA"
        debug_final_response(
            trace_id, status=status, confidence=confidence,
            answer=final_answer, sources=[v.worker_id for v in votes]
        )

        return RetrievalResult(
            trace_id=trace_id,
            final_answer=final_answer,
            confidence=confidence,
            status="SUCCESS",
            sources=[v.worker_id for v in votes],
            superposition=superposition_object
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.logger.error(f"Query handling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure pubsub is cleaned up
        if pubsub:
            try:
                pubsub.close()
            except:
                pass
