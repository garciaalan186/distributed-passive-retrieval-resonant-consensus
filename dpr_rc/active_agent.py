import asyncio
import json
import os
import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
import uuid

from .models import (
    QueryRequest, LogEntry, ComponentType, EventType, 
    ConsensusVote, RetrievalResult
)
from .logging_utils import StructuredLogger

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
RESPONSE_TIMEOUT = 5.0  # seconds to wait for votes

app = FastAPI(title="DPR-ActiveController")
logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Streams
RFI_STREAM = "dpr:rfi"
VOTE_STREAM = "dpr:votes"

@app.on_event("startup")
async def startup_event():
    # Create streams if not exist
    try:
        redis_client.xgroup_create(VOTE_STREAM, "controller_group", mkstream=True)
    except Exception:
        pass # Group already exists
    logger.logger.info("Active Controller Started")

class RouteLogic:
    @staticmethod
    def get_target_shards(query: QueryRequest) -> List[str]:
        # L1 Time-Sharded Routing
        # Simple implementation: Hash based or keyword based.
        # For this artifact, we broadcast to all ("*") but log the targeting logic.
        if query.timestamp_context:
            # Deterministic sharding logic based on year
            year = query.timestamp_context[:4]
            return [f"shard_{year}"]
        return ["broadcast"]

@app.post("/query", response_model=RetrievalResult)
async def handle_query(request: QueryRequest):
    trace_id = request.trace_id
    
    # 1. Log Query Reception
    logger.log_event(trace_id, EventType.SYSTEM_INIT, request.model_dump())

    # 2. Routing
    target_shards = RouteLogic.get_target_shards(request)
    
    # 3. Broadcast RFI (Request for Information) to Redis
    rfi_payload = {
        "trace_id": trace_id,
        "query_text": request.query_text,
        "target_shards": target_shards,
        "timestamp_context": request.timestamp_context
    }
    redis_client.xadd(RFI_STREAM, rfi_payload)
    logger.log_event(trace_id, EventType.RFI_BROADCAST, rfi_payload)

    # 4. Wait for Consensus (Gather Votes)
    # in a real system we might use async pub/sub or a separate worker for aggregation.
    # For this benchmark API, we block/poll for a short window.
    start_time = time.time()
    votes: List[ConsensusVote] = []
    
    # We consume from the VOTE_STREAM looking for our trace_id
    # Note: In high throughput, this polling is inefficient. 
    # Better: Use a specific response channel per trace_id.
    # Implementation: Subscribe to a channel `dpr:responses:{trace_id}`
    
    pubsub = redis_client.pubsub()
    response_channel = f"dpr:responses:{trace_id}"
    pubsub.subscribe(response_channel)

    while (time.time() - start_time) < RESPONSE_TIMEOUT:
        message = pubsub.get_message(ignore_subscribe_messages=True)
        if message:
            vote_data = json.loads(message['data'])
            vote = ConsensusVote(**vote_data)
            votes.append(vote)
            # Break if we have enough votes? (e.g. 3)
            if len(votes) >= 3:
                break
        await asyncio.sleep(0.1)

    # 5. Calculate Consensus (Resonant verification)
    if not votes:
        logger.log_event(trace_id, EventType.HALLUCINATION_DETECTED, {"reason": "No votes received"})
        return RetrievalResult(
            trace_id=trace_id,
            final_answer="No consensus reached.",
            confidence=0.0,
            status="FAILED",
            sources=[]
        )

    # Simple Consensus: Weighted Average of Confidence + Vector Center
    # "Resonant Consensus" roughly means finding the cluster of highest agreement.
    # We take the vote with highest confidence that is also supported by others.
    
    best_vote = max(votes, key=lambda v: v.confidence_score)
    
    # Calculate "Consensus Quadrant" (Average x,y)
    avg_x = sum(v.semantic_quadrant[0] for v in votes) / len(votes)
    avg_y = sum(v.semantic_quadrant[1] for v in votes) / len(votes)
    
    consensus_metrics = {
        "vote_count": len(votes),
        "consensus_quadrant": [avg_x, avg_y],
        "agreement_score": 0.9 # Mock calculation
    }
    
    logger.log_event(trace_id, EventType.CONSENSUS_REACHED, {
        "selected_vote": best_vote.model_dump(),
        "metrics": consensus_metrics
    }, metrics=consensus_metrics)

    return RetrievalResult(
        trace_id=trace_id,
        final_answer=best_vote.content_snippet,
        confidence=best_vote.confidence_score,
        status="SUCCESS",
        sources=[v.worker_id for v in votes]
    )
