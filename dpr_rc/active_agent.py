import asyncio
import json
import os
import time
from typing import List, Dict
from contextlib import asynccontextmanager
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

logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Streams
RFI_STREAM = "dpr:rfi"
VOTE_STREAM = "dpr:votes"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        redis_client.xgroup_create(VOTE_STREAM, "controller_group", mkstream=True)
    except Exception:
        pass  # Group already exists
    logger.logger.info("Active Controller Started")
    yield
    # Shutdown (if needed)
    logger.logger.info("Active Controller Shutting Down")

app = FastAPI(title="DPR-ActiveController", lifespan=lifespan)

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
        "target_shards": json.dumps(target_shards),
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

# ... imports ...
from dataclasses import dataclass
import numpy as np

# Mocking FrozenAgentState for prototype (Architecture requirement)
class FrozenAgentState:
    def __init__(self, creation_time):
        self.creation_time = creation_time

    def verify(self, query: str, candidate: str) -> dict:
        # Simulate temporal understanding evolution
        # In a real system, this runs inference on the frozen model checkpoint
        return {
            "score": 0.9, # Mock high score
            "prompt": f"Verify: {candidate}",
            "response": "Consistent with 2020 knowledge",
            "tokens": 10
        }

@dataclass
class QuadrantClassification:
    name: str
    reasoning: str

# ... existing code ...

@app.post("/query", response_model=RetrievalResult)
async def handle_query(request: QueryRequest):
    trace_id = request.trace_id
    logger.log_event(trace_id, EventType.SYSTEM_INIT, request.model_dump())

    # 1. Broadcast RFI
    target_shards = RouteLogic.get_target_shards(request)
    rfi_payload = {
        "trace_id": trace_id,
        "query_text": request.query_text,
        "target_shards": json.dumps(target_shards),
        "timestamp_context": request.timestamp_context
    }
    redis_client.xadd(RFI_STREAM, rfi_payload)
    
    # 2. Gather Votes (L2 + L3)
    start_time = time.time()
    votes: List[ConsensusVote] = []
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"dpr:responses:{trace_id}")

    while (time.time() - start_time) < RESPONSE_TIMEOUT:
        message = pubsub.get_message(ignore_subscribe_messages=True)
        if message:
            vote_data = json.loads(message['data'])
            votes.append(ConsensusVote(**vote_data))
            if len(votes) >= 3: # Min quorum
                break
        await asyncio.sleep(0.1)

    if not votes:
        return RetrievalResult(trace_id=trace_id, final_answer="", confidence=0.0, status="FAILED", sources=[])

    # 3. Resonant Consensus Protocol (Architecture Phase L3)
    # Classify candidates into Semantic Quadrants based on historical snapshots
    
    # Group votes by content hash
    unique_candidates = {}
    for v in votes:
        if v.content_hash not in unique_candidates:
            unique_candidates[v.content_hash] = {
                "content": v.content_snippet,
                "votes": []
            }
        unique_candidates[v.content_hash]["votes"].append(v)
    
    consensus_set = []
    perspectival_set = []
    
    for chash, data in unique_candidates.items():
        # Evaluate consensus strength
        scores = [v.confidence_score for v in data["votes"]]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Quadrant Classification logic
        if mean_score > 0.7 and std_score < 0.2:
            quadrant = "SYMMETRIC_RESONANCE"
            consensus_set.append(data["content"])
        elif mean_score > 0.4: # Lower threshold to catch more perspectives
            quadrant = "ASYMMETRIC"
            perspectival_set.append({
                "claim": data["content"],
                "snapshot_views": {v.worker_id: v.confidence_score for v in data["votes"]},
                "quadrant": quadrant,
                "metrics": {"mean": mean_score, "std": std_score}
            })
        else:
             quadrant = "DISSONANT_POLARIZATION"
             # Still include significantly dissonant claims as perspectives if they have some support
             if mean_score > 0.3:
                 perspectival_set.append({
                    "claim": data["content"],
                    "snapshot_views": {v.worker_id: v.confidence_score for v in data["votes"]},
                    "quadrant": quadrant,
                    "metrics": {"mean": mean_score, "std": std_score}
                })

    # 4. Superposition Injection
    # Always include both sets to allow A* to see the full semantic quadrant
    superposition_object = {
        "consensus_facts": consensus_set,
        "perspectival_claims": perspectival_set
    }
    
    # 5. Generate Response (A*)
    # Construct prompt with superposition, explicitly instructing to present options under uncertainty
    a_star_prompt = f"""You are answering a query using retrieved memories.

CONSENSUS FACTS (strong agreement across historical versions):
{json.dumps(consensus_set, indent=2)}

EVOLVING/DIVERGENT PERSPECTIVES (varying agreement or evolution over time):
{json.dumps(perspectival_set, indent=2)}

QUERY: {request.query_text}

Instructions:
1. State any CONSENSUS FACTS as established knowledge.
2. If there are DIVERGENT PERSPECTIVES or weak consensus, state your uncertainty clearly.
3. Present the valid options/perspectives found in the provided data. Give context for each (e.g. "Some sources suggest X, while others indicate Y").
4. If options conflict, present them as alternatives for the user to validate.
5. Do not hallucinate information not present in the provided facts or perspectives.

ANSWER:"""

    # Mock A* generation logic for the benchmark
    # In a real system this would call the LLM with a_star_prompt
    
    if consensus_set:
        final_answer = " ".join(consensus_set)
        if perspectival_set:
            final_answer += "\n\nAdditionally, there are evolving perspectives: " + "; ".join([p['claim'] for p in perspectival_set])
        confidence = 0.95
    elif perspectival_set:
        # Uncertainty case: Present options
        options = [f"- {p['claim']} (Agreement: {p['metrics']['mean']:.2f})" for p in perspectival_set]
        final_answer = f"The historical record is not unified on this. Here are the perspectives found:\n" + "\n".join(options)
        confidence = 0.7 # Lower confidence due to lack of consensus, but non-zero because we have retrieval
    else:
         final_answer = "No relevant information found in the historical record."
         confidence = 0.0

    logger.log_event(trace_id, EventType.CONSENSUS_REACHED, {
        "superposition": superposition_object,
        "final_answer": final_answer,
        "prompt": a_star_prompt
    })

    return RetrievalResult(
        trace_id=trace_id,
        final_answer=final_answer,
        confidence=confidence,
        status="SUCCESS",
        sources=[v.worker_id for v in votes]
    )
