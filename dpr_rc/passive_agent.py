"""
Passive Agent Worker for DPR-RC System

Per Architecture Spec Section 2.1:
"Passive Agent (P) - A serverless worker representing a frozen snapshot of history,
performing verification on demand."

LAZY LOADING ARCHITECTURE:
- Workers start with EMPTY vector stores
- When an RFI arrives with target_shards, worker loads that shard on-demand
- Shards are cached locally after first load (no re-download)
- Pre-computed embeddings are loaded from local dataset (no embedding computation at runtime)

REFACTORED ARCHITECTURE (SOLID):
This file is now a THIN FACADE - it only handles:
- FastAPI endpoint setup
- HTTP request/response handling
- Delegation to application layer use cases

All business logic is in:
- dpr_rc/domain/passive_agent/ - Domain entities and services
- dpr_rc/application/passive_agent/ - Use cases and DTOs
- dpr_rc/infrastructure/passive_agent/ - Repositories, clients, factory
"""

import os
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from .models import EventType
from .logging_utils import StructuredLogger, ComponentType
from .debug_utils import (
    debug_rfi_received,
    debug_vote_created,
)
from .infrastructure.passive_agent import PassiveAgentFactory
from .application.passive_agent import ProcessRFIRequest, ProcessRFIResponse

# Configuration
WORKER_ID = os.getenv("HOSTNAME", f"worker-{os.getpid()}")
WORKER_EPOCH = os.getenv("WORKER_EPOCH", "2020")  # Default epoch
HISTORY_BUCKET = os.getenv("HISTORY_BUCKET", None)
HISTORY_SCALE = os.getenv("HISTORY_SCALE", "medium")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CLUSTER_ID = os.getenv("CLUSTER_ID", "cluster-alpha")

logger = StructuredLogger(ComponentType.PASSIVE_WORKER)

# Global use case instance
_use_case: Optional[Any] = None


def get_use_case():
    """Lazy initialize use case from factory."""
    global _use_case
    if _use_case is None:
        _use_case = PassiveAgentFactory.create_from_env()
    return _use_case


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    logger.logger.info(f"Passive Worker {WORKER_ID} started (HTTP mode, lazy loading)")
    yield
    logger.logger.info(f"Passive Worker {WORKER_ID} shutting down")


app = FastAPI(title="DPR-PassiveWorker", lifespan=lifespan)


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint"""
    use_case = get_use_case()

    # Get loaded shards from repository
    loaded_shards = []
    if use_case and hasattr(use_case, 'shard_repository'):
        try:
            # Access the loaded shards from the repository
            loaded_shards = list(use_case.shard_repository._loaded_shards.keys())
        except Exception:
            loaded_shards = []

    return {
        "status": "healthy",
        "worker_id": WORKER_ID,
        "epoch": WORKER_EPOCH,
        "mode": "lazy_loading",
        "loaded_shards": loaded_shards,
        "bucket": HISTORY_BUCKET,
        "scale": HISTORY_SCALE,
        "model": EMBEDDING_MODEL,
    }


@app.get("/shards")
def list_shards():
    """List loaded shards and their document counts"""
    use_case = get_use_case()
    shards = {}

    if use_case and hasattr(use_case, 'shard_repository'):
        try:
            for shard_id in use_case.shard_repository._loaded_shards.keys():
                shard_data = use_case.shard_repository.get_shard_data(shard_id)
                if shard_data:
                    shards[shard_id] = {
                        "count": shard_data.get("document_count", 0),
                        "loaded_from": shard_data.get("loaded_from", "unknown"),
                    }
        except Exception:
            pass

    return {"shards": shards}


class RFIRequest(BaseModel):
    """Request model for HTTP-based RFI processing"""

    trace_id: str
    query_text: str
    original_query: Optional[str] = None
    target_shards: List[str] = []
    timestamp_context: Optional[str] = None


class VoteResponse(BaseModel):
    """Response model for HTTP-based RFI processing"""

    worker_id: str
    votes: List[Dict[str, Any]]
    shards_queried: List[str]


@app.post("/process_rfi", response_model=VoteResponse)
def process_rfi_http(request: RFIRequest):
    """
    HTTP endpoint for processing RFI requests directly.

    Returns votes synchronously.
    """
    use_case = get_use_case()

    # Log RFI receipt
    logger.log_message(
        trace_id=request.trace_id,
        direction="request",
        message_type="rfi_received",
        payload=request.model_dump(),
        metadata={"worker_id": WORKER_ID, "cluster": CLUSTER_ID},
    )

    # DEBUG: RFI received
    debug_rfi_received(
        request.trace_id, WORKER_ID, request.model_dump()
    )

    # Convert to DTO
    rfi_request = ProcessRFIRequest(
        trace_id=request.trace_id,
        query_text=request.query_text,
        original_query=request.original_query or request.query_text,
        timestamp_context=request.timestamp_context or "",
        target_shards=request.target_shards,
    )

    # Execute use case
    response = use_case.execute(rfi_request)

    # Log vote response
    response_data = {
        "worker_id": WORKER_ID,
        "votes": response.votes,
        "vote_count": len(response.votes),
    }
    logger.log_message(
        trace_id=request.trace_id,
        direction="response",
        message_type="vote_response",
        payload=response_data,
        metadata={
            "vote_count": len(response.votes),
            "shards_queried": request.target_shards or [f"shard_{WORKER_EPOCH}"],
        },
    )

    # DEBUG: Votes created
    for vote in response.votes:
        debug_vote_created(WORKER_ID, request.trace_id, vote)

    # Return votes as HTTP response
    return VoteResponse(
        worker_id=WORKER_ID,
        votes=response.votes,
        shards_queried=request.target_shards or [f"shard_{WORKER_EPOCH}"],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
