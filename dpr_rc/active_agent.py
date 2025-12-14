import asyncio
import json
import os
import time
import math
from typing import List, Dict
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

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
RESPONSE_TIMEOUT = float(os.getenv("RESPONSE_TIMEOUT", "5.0"))  # seconds to wait for votes
MIN_VOTES_FOR_CONSENSUS = int(os.getenv("MIN_VOTES", "1"))  # Minimum votes needed

logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Streams
RFI_STREAM = "dpr:rfi"
VOTE_STREAM = "dpr:votes"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    # Startup
    try:
        redis_client.xgroup_create(VOTE_STREAM, "controller_group", mkstream=True)
    except Exception:
        pass  # Group already exists
    logger.logger.info("Active Controller Started")
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
    return {"status": "healthy", "component": "active_controller"}


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

    pubsub = None
    try:
        # 1. L1 Routing - Determine target shards
        target_shards = RouteLogic.get_target_shards(request)

        # 2. CRITICAL FIX: Subscribe to response channel BEFORE broadcasting RFI
        # Per Redis Pub/Sub semantics, messages are NOT queued for late subscribers.
        # We must subscribe first to ensure we don't miss any votes.
        pubsub = redis_client.pubsub()
        response_channel = f"dpr:responses:{trace_id}"
        pubsub.subscribe(response_channel)

        # Small delay to ensure subscription is fully active
        await asyncio.sleep(0.05)

        # 3. Broadcast RFI (Request for Information) to Redis Stream
        # Per Spec Section 4.2: "A* publishes an RFI message to Redis"
        rfi_payload = {
            "trace_id": trace_id,
            "query_text": request.query_text,
            "target_shards": json.dumps(target_shards),
            "timestamp_context": request.timestamp_context or ""
        }
        redis_client.xadd(RFI_STREAM, rfi_payload)
        logger.log_event(trace_id, EventType.RFI_BROADCAST, rfi_payload)

        # 4. Gather Votes (L2 + L3) - Wait for Passive Agent responses
        # Per Spec Section 4.3: "Targeted Passive Agents wake up and ingest the query"
        start_time = time.time()
        votes: List[ConsensusVote] = []

        while (time.time() - start_time) < RESPONSE_TIMEOUT:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message and message.get('data'):
                try:
                    # Handle both string and bytes data
                    data = message['data']
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    vote_data = json.loads(data)
                    votes.append(ConsensusVote(**vote_data))

                    # Check if we have minimum quorum
                    if len(votes) >= MIN_VOTES_FOR_CONSENSUS:
                        # Wait a bit more for additional votes
                        await asyncio.sleep(0.3)
                        # Drain any remaining messages
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
