"""
Active Agent Controller for DPR-RC System

REFACTORED ARCHITECTURE (SOLID):
This file is now a THIN FACADE - it only handles:
- FastAPI endpoint setup
- HTTP request/response handling
- Delegation to application layer use cases

All business logic is in:
- dpr_rc/domain/active_agent/ - Domain entities and services
- dpr_rc/application/active_agent/ - Use cases and DTOs
- dpr_rc/infrastructure/active_agent/ - Repositories, clients, factory
"""

import os
from typing import Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI
from redis import Redis

from .models import QueryRequest, RetrievalResult
from .logging_utils import StructuredLogger, ComponentType
from .debug_utils import (
    debug_query_received,
    debug_query_enhancement,
    debug_routing,
    debug_consensus_calculation,
    debug_final_response,
)
from .infrastructure.active_agent import ActiveAgentFactory
from .application.active_agent import QueryRequestDTO

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SLM_SERVICE_URL = os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
PASSIVE_WORKER_URL = os.getenv("PASSIVE_WORKER_URL", "")
USE_HTTP_WORKERS = os.getenv("USE_HTTP_WORKERS", "true").lower() == "true"
ENABLE_QUERY_ENHANCEMENT = os.getenv("ENABLE_QUERY_ENHANCEMENT", "true").lower() == "true"

logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

# Initialize Redis client (may fail in HTTP-only mode)
try:
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception as e:
    logger.logger.warning(f"Redis not available: {e}. Using HTTP-only mode.")
    redis_client = None
    REDIS_AVAILABLE = False

# Global use case instance
_use_case: Optional[Any] = None


def get_use_case():
    """Lazy initialize use case from factory."""
    global _use_case
    if _use_case is None:
        _use_case = ActiveAgentFactory.create_from_env()
    return _use_case


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    logger.logger.info(
        f"Active Controller Started (Redis: {REDIS_AVAILABLE}, HTTP Workers: {USE_HTTP_WORKERS})"
    )
    yield
    logger.logger.info("Active Controller Shutting Down")


app = FastAPI(title="DPR-ActiveController", lifespan=lifespan)


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
        "passive_worker_url": PASSIVE_WORKER_URL,
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
            "negative_consensus": [],
        },
    )


@app.post("/query", response_model=RetrievalResult)
async def handle_query(request: QueryRequest):
    """
    Main query handler implementing the DPR Protocol.

    Flow (delegated to use case):
    1. Query Enhancement (L0) - SLM improves retrieval quality
    2. Gap Detection & Routing (L1) - Time-based shard selection
    3. Targeted RFI Broadcast - Request For Information to workers
    4. Gather Votes (L2) - Verification results from workers
    5. Resonant Consensus (L3) - RCP v4 classification
    6. Superposition Injection - Return all possible states
    """
    use_case = get_use_case()

    # DEBUG: Query received
    debug_query_received(request.trace_id, request.query_text, request.timestamp_context)

    # Convert to DTO
    request_dto = QueryRequestDTO(
        trace_id=request.trace_id,
        query_text=request.query_text,
        timestamp_context=request.timestamp_context,
    )

    # Execute use case
    response_dto = use_case.execute(request_dto)

    # DEBUG: Final response
    debug_final_response(
        response_dto.trace_id,
        status=response_dto.status,
        confidence=response_dto.confidence or 0.0,
        answer=response_dto.final_answer or "No consensus",
        sources=response_dto.sources,
    )

    # Convert to FastAPI response model
    return RetrievalResult(
        trace_id=response_dto.trace_id,
        final_answer=response_dto.final_answer,
        confidence=response_dto.confidence,
        status=response_dto.status,
        sources=response_dto.sources,
        superposition=response_dto.superposition,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
