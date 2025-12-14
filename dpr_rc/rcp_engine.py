"""
Resonant Consensus Protocol (RCP) Engine

Per Architecture Spec Section 4.4 and Mathematical Model Section 6.2:
Computes semantic quadrant topology from peer votes to identify
symmetric resonance (consensus) and asymmetric perspectives.

This module handles:
1. Collecting responses from Redis cache
2. Collecting peer votes from Redis cache
3. Computing semantic quadrant coordinates
4. Writing final RCP result to Redis cache
5. Notifying Active Agent via Redis Stream
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from statistics import mean
import redis

from .models import (
    ComponentType, EventType, CachedResponse, PeerVote,
    AgentResponseScore, SemanticQuadrant, RCPResult
)
from .logging_utils import StructuredLogger

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# TTLs
RESPONSE_TTL = 60  # seconds
VOTE_TTL = 60  # seconds
RESULT_TTL = 300  # seconds

# Streams
RESULTS_NOTIFY_STREAM = "dpr:results:notify"

logger = StructuredLogger(ComponentType.RCP_ENGINE)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class RCPEngine:
    """
    Resonant Consensus Protocol Engine.

    Computes semantic quadrant topology from passive agent responses
    and peer votes, following the mathematical model specification.
    """

    def __init__(self, redis_client_override=None):
        self.redis = redis_client_override or redis_client

    def get_cached_responses(self, trace_id: str) -> List[CachedResponse]:
        """
        Retrieve all cached responses for a trace_id.
        Pattern: dpr:response:{trace_id}:*
        """
        responses = []
        pattern = f"dpr:response:{trace_id}:*"

        for key in self.redis.scan_iter(match=pattern):
            data = self.redis.get(key)
            if data:
                try:
                    responses.append(CachedResponse(**json.loads(data)))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.logger.warning(f"Failed to parse response {key}: {e}")

        return responses

    def get_peer_votes(self, trace_id: str) -> List[PeerVote]:
        """
        Retrieve all peer votes for a trace_id.
        Pattern: dpr:vote:{trace_id}:*:*
        """
        votes = []
        pattern = f"dpr:vote:{trace_id}:*"

        for key in self.redis.scan_iter(match=pattern):
            data = self.redis.get(key)
            if data:
                try:
                    votes.append(PeerVote(**json.loads(data)))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.logger.warning(f"Failed to parse vote {key}: {e}")

        return votes

    def compute_semantic_quadrant(
        self,
        responses: List[CachedResponse],
        votes: List[PeerVote]
    ) -> Tuple[SemanticQuadrant, Dict[str, AgentResponseScore]]:
        """
        Compute semantic quadrant topology from responses and votes.

        Per Mathematical Model Section 6.2:
        "We map each response r_k to a topological coordinate ⟨v+, v−⟩ based on
        the net alignment of positive and negative clusters within the voting population."

        v+ = agreement score (symmetric resonance)
        v- = disagreement score (polarization)
        """
        pa_scores: Dict[str, AgentResponseScore] = {}

        for response in responses:
            agent_id = response.agent_id

            # Get all votes for this agent's response
            votes_for_agent = [v for v in votes if v.votee_id == agent_id]

            if votes_for_agent:
                # Aggregate agreement and disagreement scores
                v_plus = mean([v.agreement_score for v in votes_for_agent])
                v_minus = mean([v.disagreement_score for v in votes_for_agent])
            else:
                # No peer votes - use self-confidence as baseline
                v_plus = response.confidence
                v_minus = 1.0 - response.confidence

            pa_scores[agent_id] = AgentResponseScore(
                content=response.content or "",
                confidence=response.confidence,
                consensus_score=v_plus,
                polarization_score=v_minus,
                shard_id=response.shard_id,
                quadrant_coords=[round(v_plus, 3), round(v_minus, 3)]
            )

        # Classify into quadrants
        # Symmetric Resonance: high agreement (v+ > 0.7), low polarization (v- < 0.3)
        symmetric_responses = [
            pa_scores[aid] for aid in pa_scores
            if pa_scores[aid].consensus_score > 0.7 and pa_scores[aid].polarization_score < 0.3
        ]

        # Asymmetric Perspectives: high polarization (v- > 0.5) but still confident
        asymmetric_responses = [
            pa_scores[aid] for aid in pa_scores
            if pa_scores[aid].polarization_score > 0.5
        ]

        # Build symmetric resonance result
        if symmetric_responses:
            # Aggregate content from high-consensus responses
            aggregated_content = self._aggregate_content(symmetric_responses)
            avg_confidence = mean([r.confidence for r in symmetric_responses])
            symmetric_resonance = {
                "content": aggregated_content,
                "confidence": round(avg_confidence, 3),
                "num_sources": len(symmetric_responses)
            }
        else:
            symmetric_resonance = {
                "content": "",
                "confidence": 0.0,
                "num_sources": 0
            }

        # Build asymmetric perspectives list
        asymmetric_perspectives = [
            {
                "agent_id": r.shard_id,  # Use shard_id for identification
                "content": r.content[:200] if r.content else "",
                "polarization_score": r.polarization_score,
                "confidence": r.confidence
            }
            for r in asymmetric_responses
        ]

        semantic_quadrant = SemanticQuadrant(
            symmetric_resonance=symmetric_resonance,
            asymmetric_perspectives=asymmetric_perspectives
        )

        return semantic_quadrant, pa_scores

    def _aggregate_content(self, responses: List[AgentResponseScore]) -> str:
        """
        Aggregate content from multiple high-consensus responses.
        Returns a combined summary prioritizing unique information.
        """
        if not responses:
            return ""

        # Sort by confidence and take unique content
        sorted_responses = sorted(responses, key=lambda r: r.confidence, reverse=True)

        seen_content = set()
        aggregated_parts = []

        for r in sorted_responses:
            # Simple deduplication by first 100 chars
            content_key = r.content[:100] if r.content else ""
            if content_key not in seen_content and r.content:
                seen_content.add(content_key)
                aggregated_parts.append(r.content)

        return " ".join(aggregated_parts[:3])  # Limit to top 3 unique responses

    def compute_and_cache_result(
        self,
        trace_id: str,
        expected_responses: int,
        timeout: float = 5.0
    ) -> Optional[RCPResult]:
        """
        Wait for responses and votes, compute RCP result, and cache it.

        Args:
            trace_id: The trace ID to compute results for
            expected_responses: Number of expected agent responses (i)
            timeout: Maximum time to wait for all responses and votes

        Returns:
            RCPResult if successful, None if timeout or insufficient data
        """
        start_time = time.time()
        expected_votes = expected_responses * (expected_responses - 1)  # i × (i-1)

        responses = []
        votes = []

        # Wait for responses and votes
        while (time.time() - start_time) < timeout:
            responses = self.get_cached_responses(trace_id)
            votes = self.get_peer_votes(trace_id)

            responses_ready = len(responses) >= expected_responses
            votes_ready = len(votes) >= expected_votes or expected_responses <= 1

            if responses_ready and votes_ready:
                break

            time.sleep(0.1)

        if not responses:
            logger.logger.warning(f"No responses received for trace {trace_id}")
            return None

        # Compute semantic quadrant
        computation_start = time.time()
        semantic_quadrant, pa_scores = self.compute_semantic_quadrant(responses, votes)
        computation_time = (time.time() - computation_start) * 1000

        # Build result
        result = RCPResult(
            trace_id=trace_id,
            semantic_quadrant=semantic_quadrant,
            pa_response_scores=pa_scores,
            total_responses=len(responses),
            total_votes=len(votes),
            computation_time_ms=round(computation_time, 2)
        )

        # Cache result
        result_key = f"dpr:result:{trace_id}"
        self.redis.setex(result_key, RESULT_TTL, result.model_dump_json())

        # Notify via stream
        self.redis.xadd(RESULTS_NOTIFY_STREAM, {
            "trace_id": trace_id,
            "status": "complete",
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.log_event(trace_id, EventType.CONSENSUS_REACHED, {
            "total_responses": len(responses),
            "total_votes": len(votes),
            "symmetric_sources": semantic_quadrant.symmetric_resonance.get("num_sources", 0),
            "asymmetric_count": len(semantic_quadrant.asymmetric_perspectives)
        })

        return result

    def get_cached_result(self, trace_id: str) -> Optional[RCPResult]:
        """
        Retrieve cached RCP result if available.
        """
        result_key = f"dpr:result:{trace_id}"
        data = self.redis.get(result_key)

        if data:
            try:
                return RCPResult(**json.loads(data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.logger.warning(f"Failed to parse result {result_key}: {e}")

        return None


# Module-level functions for convenience

def wait_for_rcp_result(trace_id: str, timeout: float = 5.0) -> Optional[RCPResult]:
    """
    Wait for RCP result to be available in cache.
    Used by Active Agent to poll for results.
    """
    engine = RCPEngine()
    deadline = time.time() + timeout

    while time.time() < deadline:
        result = engine.get_cached_result(trace_id)
        if result:
            return result
        time.sleep(0.1)

    return None


def trigger_rcp_computation(trace_id: str, expected_agents: int) -> Optional[RCPResult]:
    """
    Trigger RCP computation for a trace.
    Can be called by any component to initiate consensus computation.
    """
    engine = RCPEngine()
    return engine.compute_and_cache_result(trace_id, expected_agents)
