"""
ProcessQueryUseCase

Core use case for processing queries through the DPR-RC system.

This use case encapsulates the main DPR Protocol flow:
1. Query Enhancement (via SLM)
2. L1 Routing (determine target shards)
3. Gather Votes (from passive workers)
4. Resonant Consensus (L3)
5. Superposition Injection

IMPORTANT: This is the EXACT same logic as in active_agent.py's handle_query,
but extracted into a reusable use case. This ensures benchmarks test the same
code path as production HTTP endpoints.
"""

import math
from typing import List, Dict, Optional

from dpr_rc.models import ConsensusVote
from dpr_rc.application.dtos import ProcessQueryRequest, ProcessQueryResponse
from dpr_rc.application.interfaces import ISLMService, IRouterService, IWorkerService
from dpr_rc.logging_utils import StructuredLogger, ComponentType, EventType
from dpr_rc.debug_utils import (
    debug_query_received, debug_query_enhancement, debug_routing,
    debug_consensus_calculation, debug_final_response
)
from dpr_rc.config import get_dpr_config

# Load consensus config
_consensus_config = get_dpr_config().get('consensus', {})


class ProcessQueryUseCase:
    """
    Use case for processing queries through the DPR-RC system.

    This use case implements the core DPR Protocol (Spec Section 4):
    1. Gap Detection & Routing (L1)
    2. Query Enhancement (SLM)
    3. Targeted RFI Broadcast
    4. Gather Votes (L2 Verification results)
    5. Resonant Consensus (L3)
    6. Superposition Injection

    Design Principles:
    - Pure business logic (no HTTP, no framework dependencies)
    - Dependency injection for all external services
    - Same logic as production HTTP endpoint
    - Testable in isolation
    """

    def __init__(
        self,
        slm_service: ISLMService,
        router_service: IRouterService,
        worker_service: IWorkerService,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the use case with injected dependencies.

        Args:
            slm_service: Service for query enhancement
            router_service: Service for L1 routing decisions
            worker_service: Service for worker communication
            logger: Optional structured logger (creates default if None)
        """
        self._slm_service = slm_service
        self._router_service = router_service
        self._worker_service = worker_service
        self._logger = logger or StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

    async def execute(self, request: ProcessQueryRequest) -> ProcessQueryResponse:
        """
        Execute the query processing use case.

        This method contains the EXACT same logic as active_agent.py's handle_query,
        ensuring benchmarks test the same code path as production.

        Args:
            request: ProcessQueryRequest DTO

        Returns:
            ProcessQueryResponse DTO with results

        Note:
            This method never raises exceptions. All errors are captured
            in the response with appropriate status codes.
        """
        trace_id = request.trace_id
        self._logger.log_event(trace_id, EventType.SYSTEM_INIT, {
            "query_text": request.query_text,
            "timestamp_context": request.timestamp_context
        })

        # DEBUG: Query received
        debug_query_received(trace_id, request.query_text, request.timestamp_context)

        try:
            # 0. Query Enhancement via SLM
            # The SLM expands abbreviations, adds synonyms, and clarifies ambiguous terms
            enhancement_result = self._slm_service.enhance_query(
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

            self._logger.log_event(trace_id, EventType.SYSTEM_INIT, {
                "original_query": request.query_text,
                "enhanced_query": enhanced_query,
                "expansions": enhancement_result.get("expansions", []),
                "enhancement_used": enhancement_result.get("enhancement_used", False)
            })

            # 1. L1 Routing - Determine target shards
            target_shards = self._router_service.get_target_shards(
                enhanced_query,
                request.timestamp_context
            )

            # DEBUG: Routing decision
            debug_routing(trace_id, enhanced_query, target_shards)

            # 2. Gather Votes from passive workers
            votes: List[ConsensusVote] = await self._worker_service.gather_votes(
                trace_id=trace_id,
                query_text=enhanced_query,
                original_query=request.query_text,
                target_shards=target_shards,
                timestamp_context=request.timestamp_context
            )

            self._logger.log_event(trace_id, EventType.RFI_BROADCAST, {
                "target_shards": target_shards,
                "votes_received": len(votes)
            })

            # Handle no votes case
            if not votes:
                self._logger.log_event(
                    trace_id,
                    EventType.HALLUCINATION_DETECTED,
                    {"reason": "No votes received"}
                )
                # DEBUG: No votes received
                debug_final_response(
                    trace_id, status="FAILED", confidence=0.0,
                    answer="No consensus reached.", sources=[]
                )
                return ProcessQueryResponse(
                    trace_id=trace_id,
                    final_answer="No consensus reached.",
                    confidence=0.0,
                    status="FAILED",
                    sources=[],
                    metadata={
                        "enhanced_query": enhanced_query,
                        "target_shards": target_shards,
                        "num_votes": 0
                    }
                )

            # 3. Resonant Consensus Protocol (L3) - Per Spec Section 4.4
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

                # Quadrant Classification per spec using config thresholds
                # High mean + low std = strong consensus (Symmetric Resonance)
                # Per Mathematical Model: "High-entropy bridge concepts agreed upon by diverse contexts"
                mean_threshold = _consensus_config.get('mean_threshold', 0.7)
                std_threshold = _consensus_config.get('std_threshold', 0.2)
                asymmetric_threshold = _consensus_config.get('asymmetric_threshold', 0.4)
                dissonant_threshold = _consensus_config.get('dissonant_threshold', 0.3)

                if mean_score > mean_threshold and std_score < std_threshold:
                    quadrant = "SYMMETRIC_RESONANCE"
                    consensus_set.append(data["content"])
                elif mean_score > asymmetric_threshold:
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
                    if mean_score > dissonant_threshold:
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

            # 4. Superposition Injection - Per Spec Section 4.5
            # "A* injects a Superposition Object into its context, containing both
            # the Consensus Truth and distinct Perspectives"
            superposition_object = {
                "consensus_facts": consensus_set,
                "perspectival_claims": perspectival_set
            }

            # 5. Generate Response (A*) - Construct answer from superposition
            confidence_config = _consensus_config.get('confidence', {})
            consensus_confidence = confidence_config.get('consensus', 0.95)
            perspectival_confidence = confidence_config.get('perspectival', 0.7)

            if consensus_set:
                final_answer = " ".join(consensus_set)
                if perspectival_set:
                    final_answer += "\n\nAdditionally, there are evolving perspectives: " + \
                                   "; ".join([p['claim'] for p in perspectival_set])
                confidence = consensus_confidence
            elif perspectival_set:
                # Uncertainty case: Present options per spec
                # "allows A* to generate a nuanced reply acknowledging both the agreed facts
                # and the conflicting perspectives"
                options = [f"- {p['claim']} (Agreement: {p['metrics']['mean']:.2f})"
                          for p in perspectival_set]
                final_answer = "The historical record shows varying perspectives:\n" + "\n".join(options)
                confidence = perspectival_confidence
            else:
                final_answer = "No relevant information found in the historical record."
                confidence = 0.0

            self._logger.log_event(trace_id, EventType.CONSENSUS_REACHED, {
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

            return ProcessQueryResponse(
                trace_id=trace_id,
                final_answer=final_answer,
                confidence=confidence,
                status=status,
                sources=[v.worker_id for v in votes],
                superposition=superposition_object,
                metadata={
                    "enhanced_query": enhanced_query,
                    "target_shards": target_shards,
                    "num_votes": len(votes),
                    "num_consensus": len(consensus_set),
                    "num_perspectival": len(perspectival_set),
                    "enhancement_info": enhancement_result
                }
            )

        except Exception as e:
            # Ensure we never raise exceptions from the use case
            # All errors are captured in the response
            self._logger.logger.error(f"Query processing error: {e}")
            import traceback
            traceback.print_exc()

            return ProcessQueryResponse(
                trace_id=trace_id,
                final_answer="",
                confidence=0.0,
                status="ERROR",
                sources=[],
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
