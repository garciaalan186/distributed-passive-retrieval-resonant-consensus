"""
Active Agent Controller for DPR-RC System (Cache-Based Architecture)

Per Architecture Spec Section 2.1:
"Active Agent (A*) - The live interface managing immediate context."

ARCHITECTURE: Cache-Based Response Model
- Replaces Redis Pub/Sub with Redis cache polling
- Eliminates race condition (cache is persistent, Pub/Sub is fire-and-forget)
- Integrates with RCP Engine for consensus computation
- Waits for worker readiness before processing queries

Per Mathematical Model Section 4.1:
"A* represents the current, evolving state of the agent's understanding."
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis

from .models import (
    QueryRequest, LogEntry, ComponentType, EventType,
    RetrievalResult, CachedResponse, RCPResult, AgentResponseScore
)
from .logging_utils import StructuredLogger
from .rcp_engine import RCPEngine, wait_for_rcp_result

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
RESPONSE_TIMEOUT = float(os.getenv("RESPONSE_TIMEOUT", "5.0"))
MIN_RESPONSES_FOR_CONSENSUS = int(os.getenv("MIN_RESPONSES", "1"))
WORKERS_READY_KEY = "dpr:workers:ready"

logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Streams
RFI_STREAM = "dpr:rfi"
RESULTS_NOTIFY_STREAM = "dpr:results:notify"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    logger.logger.info("Active Controller Started (Cache-Based Architecture)")
    yield
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
            year = query.timestamp_context[:4]
            return [f"shard_{year}"]
        return ["broadcast"]


def get_ready_worker_count() -> int:
    """Get the number of ready workers."""
    return redis_client.scard(WORKERS_READY_KEY)


async def wait_for_workers_ready(required_k: int, timeout: float = 30.0) -> bool:
    """
    Wait for required number of workers to be ready.
    CRITICAL: All workers must be ready before benchmark execution begins.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready_count = redis_client.scard(WORKERS_READY_KEY)
        if ready_count >= required_k:
            logger.logger.info(f"All {required_k} workers ready")
            return True
        await asyncio.sleep(0.5)

    ready_count = redis_client.scard(WORKERS_READY_KEY)
    logger.logger.warning(f"Only {ready_count}/{required_k} workers ready after {timeout}s")
    return ready_count >= 1  # Continue if at least 1 worker


async def collect_cached_responses(
    trace_id: str,
    timeout: float = 5.0,
    min_responses: int = 1
) -> List[CachedResponse]:
    """
    Collect responses from Redis cache instead of Pub/Sub.

    Benefits:
    - Eliminates race condition (cache is persistent)
    - Responses can be inspected/debugged directly
    - Natural deduplication via agent_id in key
    """
    responses = []
    seen_agents = set()
    deadline = time.time() + timeout
    pattern = f"dpr:response:{trace_id}:*"

    while time.time() < deadline:
        # Scan for response keys
        for key in redis_client.scan_iter(match=pattern):
            agent_id = key.split(":")[-1]
            if agent_id not in seen_agents:
                data = redis_client.get(key)
                if data:
                    try:
                        response = CachedResponse(**json.loads(data))
                        responses.append(response)
                        seen_agents.add(agent_id)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.logger.warning(f"Failed to parse response: {e}")

        # Check if we have enough responses
        if len(responses) >= min_responses:
            # Wait a bit more for additional responses
            await asyncio.sleep(0.3)
            # Final drain
            for key in redis_client.scan_iter(match=pattern):
                agent_id = key.split(":")[-1]
                if agent_id not in seen_agents:
                    data = redis_client.get(key)
                    if data:
                        try:
                            responses.append(CachedResponse(**json.loads(data)))
                            seen_agents.add(agent_id)
                        except (json.JSONDecodeError, ValueError):
                            pass
            break

        await asyncio.sleep(0.1)

    return responses


def build_response_from_rcp(trace_id: str, rcp_result: RCPResult) -> RetrievalResult:
    """
    Build final RetrievalResult from RCP computation.
    """
    semantic_quadrant = rcp_result.semantic_quadrant

    # Build superposition object for backward compatibility
    consensus_facts = []
    perspectival_claims = []

    if semantic_quadrant.symmetric_resonance.get("content"):
        consensus_facts.append(semantic_quadrant.symmetric_resonance["content"])

    for perspective in semantic_quadrant.asymmetric_perspectives:
        perspectival_claims.append({
            "claim": perspective.get("content", ""),
            "quadrant": "ASYMMETRIC",
            "metrics": {
                "polarization": perspective.get("polarization_score", 0),
                "confidence": perspective.get("confidence", 0)
            }
        })

    superposition_object = {
        "consensus_facts": consensus_facts,
        "perspectival_claims": perspectival_claims
    }

    # Generate final answer
    if consensus_facts:
        final_answer = " ".join(consensus_facts)
        if perspectival_claims:
            final_answer += "\n\nAdditionally, there are evolving perspectives: " + \
                           "; ".join([p['claim'] for p in perspectival_claims if p['claim']])
        confidence = semantic_quadrant.symmetric_resonance.get("confidence", 0.9)
    elif perspectival_claims:
        options = [f"- {p['claim']} (Polarization: {p['metrics']['polarization']:.2f})"
                  for p in perspectival_claims if p['claim']]
        final_answer = "The historical record shows varying perspectives:\n" + "\n".join(options)
        confidence = 0.7
    else:
        final_answer = "No relevant information found in the historical record."
        confidence = 0.0

    return RetrievalResult(
        trace_id=trace_id,
        final_answer=final_answer,
        confidence=confidence,
        status="SUCCESS" if confidence > 0 else "FAILED",
        sources=list(rcp_result.pa_response_scores.keys()),
        superposition=superposition_object,
        rcp_result=rcp_result
    )


def build_response_from_cached(trace_id: str, responses: List[CachedResponse]) -> RetrievalResult:
    """
    Build RetrievalResult directly from cached responses when RCP not available.
    Fallback for single-response scenarios.
    """
    if not responses:
        return RetrievalResult(
            trace_id=trace_id,
            final_answer="No consensus reached.",
            confidence=0.0,
            status="FAILED",
            sources=[]
        )

    # Simple aggregation without full RCP
    contents = [r.content for r in responses if r.content]
    confidences = [r.confidence for r in responses]

    if contents:
        final_answer = " ".join(contents[:3])  # Top 3 responses
        avg_confidence = sum(confidences) / len(confidences)
    else:
        final_answer = "No relevant content found."
        avg_confidence = 0.0

    return RetrievalResult(
        trace_id=trace_id,
        final_answer=final_answer,
        confidence=avg_confidence,
        status="SUCCESS" if avg_confidence > 0.3 else "FAILED",
        sources=[r.agent_id for r in responses],
        superposition={
            "consensus_facts": contents,
            "perspectival_claims": []
        }
    )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    ready_workers = get_ready_worker_count()
    return {
        "status": "healthy",
        "component": "active_controller",
        "architecture": "cache-based",
        "ready_workers": ready_workers
    }


@app.get("/workers/ready")
def workers_ready():
    """Get count of ready workers"""
    count = get_ready_worker_count()
    workers = list(redis_client.smembers(WORKERS_READY_KEY))
    return {
        "ready_count": count,
        "workers": workers
    }


@app.post("/query", response_model=RetrievalResult)
async def handle_query(request: QueryRequest):
    """
    Main query handler implementing the DPR Protocol with Cache-Based Architecture.

    Flow:
    1. Check worker readiness
    2. L1 Routing - Determine target shards
    3. Broadcast RFI to Redis Stream
    4. Collect responses from Redis cache (not Pub/Sub)
    5. Trigger RCP computation or wait for result
    6. Build and return response
    """
    trace_id = request.trace_id
    logger.log_event(trace_id, EventType.SYSTEM_INIT, request.model_dump())

    try:
        # 1. Check worker readiness (at least 1 worker needed)
        ready_count = get_ready_worker_count()
        if ready_count < 1:
            logger.logger.warning(f"No workers ready, attempting anyway")

        # 2. L1 Routing - Determine target shards
        target_shards = RouteLogic.get_target_shards(request)
        expected_responses = min(ready_count, len(target_shards)) if ready_count > 0 else 1

        # 3. Broadcast RFI to Redis Stream
        rfi_payload = {
            "trace_id": trace_id,
            "query_text": request.query_text,
            "target_shards": json.dumps(target_shards),
            "timestamp_context": request.timestamp_context or ""
        }
        redis_client.xadd(RFI_STREAM, rfi_payload)
        logger.log_event(trace_id, EventType.RFI_BROADCAST, rfi_payload)

        # 4. Collect responses from Redis cache
        responses = await collect_cached_responses(
            trace_id,
            timeout=RESPONSE_TIMEOUT,
            min_responses=MIN_RESPONSES_FOR_CONSENSUS
        )

        if not responses:
            logger.log_event(trace_id, EventType.HALLUCINATION_DETECTED,
                           {"reason": "No responses received"})
            return RetrievalResult(
                trace_id=trace_id,
                final_answer="No consensus reached.",
                confidence=0.0,
                status="FAILED",
                sources=[]
            )

        # 5. Use RCP Engine for consensus computation
        rcp_engine = RCPEngine()

        # Try to compute RCP result
        rcp_result = rcp_engine.compute_and_cache_result(
            trace_id,
            expected_responses=len(responses),
            timeout=2.0  # Short timeout for computation
        )

        # 6. Build response
        if rcp_result:
            result = build_response_from_rcp(trace_id, rcp_result)
        else:
            # Fallback to simple aggregation
            result = build_response_from_cached(trace_id, responses)

        logger.log_event(trace_id, EventType.CONSENSUS_REACHED, {
            "final_answer": result.final_answer[:200],
            "num_responses": len(responses),
            "confidence": result.confidence,
            "used_rcp": rcp_result is not None
        })

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.logger.error(f"Query handling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/with_worker_check", response_model=RetrievalResult)
async def handle_query_with_worker_check(
    request: QueryRequest,
    required_workers: int = 1,
    worker_timeout: float = 10.0
):
    """
    Query handler that waits for required workers before processing.
    Use this for benchmark execution to ensure workers are ready.
    """
    # Wait for workers
    workers_ready = await wait_for_workers_ready(required_workers, worker_timeout)
    if not workers_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Insufficient workers ready. Required: {required_workers}"
        )

    # Process query
    return await handle_query(request)
