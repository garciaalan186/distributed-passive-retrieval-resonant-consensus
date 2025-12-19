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
    ConsensusVote, RetrievalResult, RCPConfig
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
SLM_ENHANCE_TIMEOUT = float(os.getenv("SLM_ENHANCE_TIMEOUT", "30.0"))  # seconds (increased for cold starts)
ENABLE_QUERY_ENHANCEMENT = os.getenv("ENABLE_QUERY_ENHANCEMENT", "true").lower() == "true"

# HTTP-based worker communication (for Cloud Run without Redis)
# Can be comma-separated list of worker URLs
PASSIVE_WORKER_URL = os.getenv("PASSIVE_WORKER_URL", "")
USE_HTTP_WORKERS = os.getenv("USE_HTTP_WORKERS", "true").lower() == "true"
HTTP_WORKER_TIMEOUT = float(os.getenv("HTTP_WORKER_TIMEOUT", "90.0"))  # seconds (increased for cold starts + GCS loading)

logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

# RCP v4: Resonant Consensus Protocol configuration
rcp_config = RCPConfig()

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
    """
    L1 Time-Sharded Routing Logic per DPR Architecture Spec Section 3.1.1

    Enhanced with tempo-normalized sharding and causal awareness.
    """

    _manifest = None
    _causal_index = None
    _manifest_loaded = False

    @classmethod
    def _load_indices(cls):
        """Load shard manifest and causal index from GCS (lazy loading)."""
        if cls._manifest_loaded:
            return

        try:
            from google.cloud import storage
            import json
            import tempfile

            bucket_name = os.environ.get('HISTORY_BUCKET', '')
            scale = os.environ.get('HISTORY_SCALE', 'medium')

            if not bucket_name:
                logger.logger.warning("HISTORY_BUCKET not set, using legacy year-based routing")
                cls._manifest_loaded = True
                return

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Load shard manifest
            manifest_blob = bucket.blob(f"indices/{scale}/shard_manifest.json")
            if manifest_blob.exists():
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
                    manifest_blob.download_to_filename(f.name)
                    f.seek(0)
                    manifest_data = json.load(open(f.name))
                    cls._manifest = manifest_data
                    logger.logger.info(f"Loaded shard manifest: {len(manifest_data.get('shards', []))} shards")

            # Load causal index
            causal_blob = bucket.blob(f"indices/{scale}/causal_index.json")
            if causal_blob.exists():
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
                    causal_blob.download_to_filename(f.name)
                    f.seek(0)
                    causal_data = json.load(open(f.name))
                    cls._causal_index = causal_data
                    logger.logger.info(f"Loaded causal index")

            cls._manifest_loaded = True

        except Exception as e:
            logger.logger.error(f"Failed to load indices: {e}, falling back to legacy routing")
            cls._manifest_loaded = True

    @classmethod
    def get_target_shards(cls, query: QueryRequest) -> List[str]:
        """
        Determine target shards based on timestamp context.

        Enhanced routing:
        1. Query manifest to find primary shard(s) for timestamp
        2. Expand to include causal ancestor shards (up to depth 2)
        3. Fallback to legacy year-based routing if indices unavailable

        Per spec: "The Central Index is partitioned into Time-Based Shards"
        """
        cls._load_indices()

        if not query.timestamp_context:
            return ["broadcast"]

        # Use tempo-normalized routing if manifest available
        if cls._manifest and 'shards' in cls._manifest:
            return cls._get_tempo_normalized_shards(query.timestamp_context)

        # Fallback to legacy year-based routing
        year = query.timestamp_context[:4]
        return [f"shard_{year}"]

    @classmethod
    def _get_tempo_normalized_shards(cls, timestamp: str) -> List[str]:
        """
        Get tempo-normalized shards for timestamp with causal expansion.

        Args:
            timestamp: ISO timestamp

        Returns:
            List of shard IDs (primary + causal ancestors)
        """
        primary_shards = []

        # Find shards containing this timestamp
        for shard in cls._manifest.get('shards', []):
            time_range = shard.get('time_range', {})
            start = time_range.get('start', '')
            end = time_range.get('end', '')

            if start <= timestamp <= end:
                primary_shards.append(shard.get('id', ''))

        if not primary_shards:
            # No shard found for timestamp, fallback to closest
            logger.logger.warning(f"No shard found for timestamp {timestamp}, using broadcast")
            return ["broadcast"]

        # Expand to include causal ancestors (if causal index available)
        if cls._causal_index and 'shard_ancestry' in cls._causal_index:
            expanded_shards = set(primary_shards)

            for shard_id in primary_shards:
                ancestry = cls._causal_index['shard_ancestry'].get(shard_id, {})
                # Include direct ancestors (depth 1)
                direct_ancestors = ancestry.get('direct_ancestors', [])
                expanded_shards.update(direct_ancestors[:3])  # Limit to top 3 ancestors

            return sorted(list(expanded_shards))

        return primary_shards


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


@app.get("/debug/sample_response", response_model=RetrievalResult)
def debug_sample_response():
    """Return sample RetrievalResult to verify serialization"""
    return RetrievalResult(
        trace_id="test-123",
        final_answer=None,
        confidence=0.0,
        status="SUCCESS",
        sources=["worker-1"],
        superposition={
            "consensus": [{"claim": "test", "score": 0.9}],
            "polar": [],
            "negative_consensus": []
        }
    )


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

            # 4. Log worker HTTP request
            logger.log_message(
                trace_id=trace_id,
                direction="request",
                message_type="worker_rfi",
                payload=request_payload,
                metadata={"worker_url": endpoint, "target_shards": target_shards}
            )

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

                # 5. Log worker HTTP response
                logger.log_message(
                    trace_id=trace_id,
                    direction="response",
                    message_type="worker_votes",
                    payload=result,
                    metadata={"worker_url": endpoint, "vote_count": len(votes_data)}
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


# ============================================================================
# RCP v4: Resonant Consensus Protocol Implementation
# ============================================================================

def compute_cluster_approval(votes_for_artifact: List[ConsensusVote], cluster_id: str) -> bool:
    """
    RCP v4 Equation 1: Cluster Approval

    Approve_i(ω) = 1 if (1/|Ci|) * Σ v(ω,a) >= θ_i
                              a∈Ci

    Returns True if the cluster approves the artifact (fraction of binary votes >= threshold)
    """
    cluster_votes = [v for v in votes_for_artifact if v.cluster_id == cluster_id]

    if not cluster_votes:
        return False

    # Count binary approvals in this cluster
    approvals = sum(v.binary_vote for v in cluster_votes)
    approval_rate = approvals / len(cluster_votes)

    return approval_rate >= rcp_config.theta


def compute_approval_set(votes_for_artifact: List[ConsensusVote]) -> set:
    """
    RCP v4 Equation 2: Approval Set

    S(ω) = {C_i ∈ C : Approve_i(ω) = 1}

    Returns the set of cluster IDs that approve this artifact
    """
    # Get all unique clusters
    all_clusters = set(v.cluster_id for v in votes_for_artifact)

    # Determine which clusters approve
    approval_set = set()
    for cluster_id in all_clusters:
        if compute_cluster_approval(votes_for_artifact, cluster_id):
            approval_set.add(cluster_id)

    return approval_set


def compute_agreement_ratio(approval_set: set, total_clusters: int) -> float:
    """
    RCP v4 Equation 3: Agreement Ratio

    ρ(ω) = |S(ω)| / n

    Returns the fraction of clusters that approve the artifact
    """
    return len(approval_set) / total_clusters if total_clusters > 0 else 0.0


def classify_artifact(agreement_ratio: float) -> str:
    """
    RCP v4 Equation 4: Tier Classification

    Tier(ω) = { Consensus          if ρ(ω) >= τ
              { Polar              if 1-τ < ρ(ω) < τ
              { Negative_Consensus if ρ(ω) <= 1-τ

    Note: Negative consensus means cross-cluster agreement that something is NOT true
    """
    tau = rcp_config.tau

    if agreement_ratio >= tau:
        return "CONSENSUS"
    elif agreement_ratio > (1 - tau):
        return "POLAR"
    else:
        return "NEGATIVE_CONSENSUS"


def compute_artifact_score(votes_for_artifact: List[ConsensusVote]) -> float:
    """
    RCP v4 Equation 5: Artifact Score

    Score(ω) = (1 / |A|-1) * Σ v(ω,a)  for a ≠ author(ω)
                              a≠author

    Returns the fraction of non-authoring agents who approved (binary votes only)
    """
    if not votes_for_artifact:
        return 0.0

    # Get author cluster (all votes for same artifact should have same author_cluster)
    author_cluster = votes_for_artifact[0].author_cluster

    # Count approvals from non-author agents (agents = workers in different clusters)
    # In DPR-RC, author_cluster is the cluster that generated the artifact
    # Non-authors are workers from ALL clusters voting on it
    non_author_votes = [v for v in votes_for_artifact if v.cluster_id != author_cluster]

    if not non_author_votes:
        # Only the author cluster voted - score is based on author cluster votes
        author_votes = [v for v in votes_for_artifact if v.cluster_id == author_cluster]
        if not author_votes:
            return 0.0
        approvals = sum(v.binary_vote for v in author_votes)
        return approvals / len(author_votes)

    # Score based on non-author approvals
    approvals = sum(v.binary_vote for v in non_author_votes)
    return approvals / len(non_author_votes)


def compute_semantic_quadrant(votes_for_artifact: List[ConsensusVote]) -> List[float]:
    """
    RCP v4: Semantic Quadrant for DPR-RC Temporal Clustering

    Returns [v+, v-] where:
    - v+ = approval rate from C_RECENT cluster (newer history)
    - v- = approval rate from C_OLDER cluster (older history)

    This creates a 2D space for the 2×2 quadrant view:
    - [1.0, 1.0] = Both temporal clusters approve strongly (Consensus)
    - [1.0, 0.0] = Only recent history (Polar+)
    - [0.0, 1.0] = Only older history (Polar-)
    - [0.0, 0.0] = Neither approves (Reject)
    """
    recent_votes = [v for v in votes_for_artifact if v.cluster_id == "C_RECENT"]
    older_votes = [v for v in votes_for_artifact if v.cluster_id == "C_OLDER"]

    # Compute approval rates (average of binary votes)
    v_plus = (sum(v.binary_vote for v in recent_votes) / len(recent_votes)) if recent_votes else 0.0
    v_minus = (sum(v.binary_vote for v in older_votes) / len(older_votes)) if older_votes else 0.0

    return [round(v_plus, 2), round(v_minus, 2)]


def enhance_query_via_slm(query_text: str, timestamp_context: Optional[str] = None, trace_id: str = None) -> dict:
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
        trace_id: Trace ID for logging correlation

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

    # Retry with exponential backoff (3 attempts: 0s, 2s, 4s delays)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 2 ** attempt  # Exponential backoff: 2s, 4s
                logger.logger.debug(f"Retrying SLM enhance_query (attempt {attempt + 1}/{max_retries}) after {delay}s delay")
                time.sleep(delay)

            # 2. Log SLM enhancement request
            request_payload = {
                "query": query_text,
                "timestamp_context": timestamp_context
            }
            if trace_id and attempt == 0:  # Only log on first attempt
                logger.log_message(
                    trace_id=trace_id,
                    direction="request",
                    message_type="slm_enhance_query",
                    payload=request_payload,
                    metadata={"slm_url": SLM_SERVICE_URL, "attempt": attempt + 1}
                )

            response = requests.post(
                f"{SLM_SERVICE_URL}/enhance_query",
                json=request_payload,
                timeout=SLM_ENHANCE_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                logger.logger.info(
                    f"Query enhanced: '{query_text}' -> '{result.get('enhanced_query', query_text)}' "
                    f"(expansions: {result.get('expansions', [])}) [attempt {attempt + 1}]"
                )

                # 3. Log SLM enhancement response
                if trace_id:
                    logger.log_message(
                        trace_id=trace_id,
                        direction="response",
                        message_type="slm_enhance_query",
                        payload=result,
                        metadata={"inference_time_ms": result.get("inference_time_ms", 0)}
                    )
                return {
                    "original_query": query_text,
                    "enhanced_query": result.get("enhanced_query", query_text),
                    "expansions": result.get("expansions", []),
                    "enhancement_used": True,
                    "inference_time_ms": result.get("inference_time_ms", 0)
                }
            elif response.status_code == 503:
                # Service unavailable (model still loading), retry
                logger.logger.warning(
                    f"SLM service not ready (503), attempt {attempt + 1}/{max_retries}"
                )
                if attempt == max_retries - 1:
                    # Last attempt failed, fall back
                    break
                continue
            else:
                # Other error codes, don't retry
                logger.logger.warning(
                    f"SLM enhance_query returned {response.status_code}, using original query"
                )
                break

        except requests.exceptions.RequestException as e:
            logger.logger.warning(f"SLM service error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                # Last attempt failed, fall back
                break
            continue

    # Fallback: return original query after all retries exhausted
    logger.logger.info(f"Using original query after {max_retries} attempts: '{query_text}'")
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

    # 1. Log client query receipt
    logger.log_message(
        trace_id=trace_id,
        direction="request",
        message_type="client_query",
        payload=request.model_dump(),
        metadata={"endpoint": "/query"}
    )

    pubsub = None
    try:
        # 0. Query Enhancement via SLM (improves retrieval quality)
        # The SLM expands abbreviations, adds synonyms, and clarifies ambiguous terms
        enhancement_result = enhance_query_via_slm(
            request.query_text,
            request.timestamp_context,
            trace_id
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
                answer="No votes received", sources=[]
            )
            # Return empty superposition when no votes
            return RetrievalResult(
                trace_id=trace_id,
                final_answer=None,
                confidence=None,
                status="FAILED",
                sources=[],
                superposition={
                    "consensus": [],
                    "polar": [],
                    "negative_consensus": []
                }
            )

        # 5. Resonant Consensus Protocol v4 (L3) - Cross-Cluster Agreement Classification
        # RCP v4: Classify artifacts based on which adversarial clusters approve

        # Group votes by content hash (same content = same artifact)
        unique_candidates: Dict[str, Dict] = {}
        for v in votes:
            if v.content_hash not in unique_candidates:
                unique_candidates[v.content_hash] = {
                    "content": v.content_snippet,
                    "votes": [],
                    "approval_set": set(),
                    "agreement_ratio": 0.0,
                    "tier": "REJECT",
                    "score": 0.0,
                    "semantic_quadrant": [0.0, 0.0]
                }
            unique_candidates[v.content_hash]["votes"].append(v)

        # Determine total number of clusters (for DPR-RC: 2 temporal clusters)
        all_clusters = set(v.cluster_id for v in votes)
        num_clusters = len(all_clusters)

        # RCP v4: Process each artifact using Equations 1-5
        consensus_set = []
        polar_set = []
        negative_consensus_set = []

        for chash, data in unique_candidates.items():
            artifact_votes = data["votes"]

            # RCP v4 Eq. 2: Compute approval set S(ω)
            approval_set = compute_approval_set(artifact_votes)
            data["approval_set"] = approval_set

            # RCP v4 Eq. 3: Compute agreement ratio ρ(ω)
            agreement_ratio = compute_agreement_ratio(approval_set, num_clusters)
            data["agreement_ratio"] = agreement_ratio

            # RCP v4 Eq. 4: Classify into tiers
            tier = classify_artifact(agreement_ratio)
            data["tier"] = tier

            # RCP v4 Eq. 5: Compute score (for ranking within tiers)
            score = compute_artifact_score(artifact_votes)
            data["score"] = score

            # RCP v4: Compute semantic quadrant [v+, v-]
            quadrant = compute_semantic_quadrant(artifact_votes)
            data["semantic_quadrant"] = quadrant

            # Categorize by tier
            artifact_summary = {
                "claim": data["content"],
                "approval_set": list(approval_set),
                "agreement_ratio": agreement_ratio,
                "tier": tier,
                "score": score,
                "quadrant": quadrant,
                "vote_count": len(artifact_votes)
            }

            if tier == "CONSENSUS":
                consensus_set.append(artifact_summary)
            elif tier == "POLAR":
                polar_set.append(artifact_summary)
            else:  # NEGATIVE_CONSENSUS
                negative_consensus_set.append(artifact_summary)

        # Sort each set by score (descending)
        consensus_set.sort(key=lambda x: x["score"], reverse=True)
        polar_set.sort(key=lambda x: x["score"], reverse=True)
        negative_consensus_set.sort(key=lambda x: x["score"], reverse=True)

        # DEBUG: Consensus calculation results
        debug_consensus_calculation(
            trace_id,
            votes_count=len(votes),
            unique_candidates=len(unique_candidates),
            consensus_set=consensus_set,
            perspectival_set=polar_set  # Polar set for backward compatibility with debug
        )

        # 6. RCP v4 Superposition Object - Structured summary for orchestrator
        # Per RCP v4 Section 7: artifacts grouped by tier and sorted by score
        superposition_object = {
            "consensus": consensus_set,
            "polar": polar_set,
            "negative_consensus": negative_consensus_set
        }

        # 7. Synthesize Human-Readable Text from Superposition
        # Generate text response for benchmark compatibility while preserving
        # superposition structure for advanced consumers (A* orchestrator)
        if consensus_set:
            # High confidence: consensus facts
            final_answer = " ".join(consensus_set)
            if polar_set:
                final_answer += "\n\nAdditionally, there are alternative perspectives: " + \
                               "; ".join([p['content'] for p in polar_set])
            confidence = 0.95
        elif polar_set:
            # Medium confidence: perspectival claims
            options = [f"- {p['content']}" for p in polar_set]
            final_answer = "The historical record shows varying perspectives:\n" + "\n".join(options)
            confidence = 0.7
        elif negative_consensus_set:
            # Low confidence: only negative consensus
            options = [f"- {p['content']}" for p in negative_consensus_set]
            final_answer = "Limited evidence found:\n" + "\n".join(options)
            confidence = 0.4
        else:
            # No data
            final_answer = "No relevant information found in the historical record."
            confidence = 0.0

        # Status is only FAILED if superposition is completely empty
        has_artifacts = bool(consensus_set or polar_set or negative_consensus_set)
        status = "SUCCESS" if has_artifacts else "FAILED"

        logger.log_event(trace_id, EventType.CONSENSUS_REACHED, {
            "superposition": superposition_object,
            "final_answer": final_answer[:200] if final_answer else None,
            "num_votes": len(votes),
            "num_consensus": len(consensus_set),
            "num_polar": len(polar_set),
            "num_negative_consensus": len(negative_consensus_set),
            "status": status,
            "confidence": confidence
        })

        # DEBUG: Log final response with synthesized text
        debug_final_response(
            trace_id, status=status, confidence=confidence,
            answer=final_answer,
            sources=[v.worker_id for v in votes]
        )

        result = RetrievalResult(
            trace_id=trace_id,
            final_answer=final_answer,  # NOW: Synthesized text for benchmark
            confidence=confidence,       # NOW: Calculated from quadrants
            status=status,
            sources=[v.worker_id for v in votes],
            superposition=superposition_object  # STILL: Raw quadrant data for A*
        )

        # 6. Log final response to client
        logger.log_message(
            trace_id=trace_id,
            direction="response",
            message_type="client_response",
            payload=result.model_dump(),
            metadata={"status": status, "vote_count": len(votes)}
        )

        return result

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
