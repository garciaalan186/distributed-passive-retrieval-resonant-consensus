"""
Use Case: Handle Query

Application layer orchestrator for query processing.
Implements the full DPR-RC pipeline: enhance → route → gather → consensus → synthesize
"""

from typing import List, Protocol, Dict, Any, Optional
from ..dtos import QueryRequestDTO, QueryResponseDTO
from dpr_rc.domain.active_agent.services import (
    RoutingService,
    ConsensusCalculator,
    ResponseSynthesizer,
    Vote,
    FoveatedRouter
)
from dpr_rc.domain.active_agent.entities import SuperpositionState


class IQueryEnhancer(Protocol):
    """Interface for query enhancement service."""

    def enhance(
        self, query_text: str, timestamp_context: Optional[str], trace_id: str
    ) -> Dict[str, Any]:
        """Enhance query using SLM."""
        ...


class IWorkerCommunicator(Protocol):
    """Interface for worker communication."""

    def gather_votes(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str],
    ) -> List[Any]:
        """Gather votes from workers."""
        ...


class ILogger(Protocol):
    """Interface for logging operations."""

    def log_message(
        self, trace_id: str, direction: str, message_type: str, payload: Dict, metadata: Dict = None
    ) -> None:
        ...

    def log_event(self, trace_id: str, event_type: str, data: Dict, metrics: Dict = None) -> None:
        ...


class HandleQueryUseCase:
    """
    Use case for handling user queries.

    Orchestrates:
    1. Query enhancement via SLM
    2. L1 routing to determine target shards
    3. Worker communication to gather votes
    4. L3 consensus calculation
    5. Response synthesis (collapse superposition)
    """

    def __init__(
        self,
        routing_service: RoutingService,
        consensus_calculator: ConsensusCalculator,
        response_synthesizer: ResponseSynthesizer,
        query_enhancer: IQueryEnhancer,
        worker_communicator: IWorkerCommunicator,
        logger: ILogger,
        foveated_router: Optional[FoveatedRouter] = None,
        enable_query_enhancement: bool = True,
        min_votes: int = 1,
    ):
        """
        Initialize use case.

        Args:
            routing_service: Service for L1 routing
            consensus_calculator: Service for L3 consensus
            response_synthesizer: Service for response synthesis
            query_enhancer: Client for query enhancement
            worker_communicator: Client for worker communication
            logger: Logger for audit trail
            foveated_router: Service for semantic routing (optional)
            enable_query_enhancement: Whether to enhance queries
            min_votes: Minimum votes needed for consensus
        """
        self.routing_service = routing_service
        self.consensus_calculator = consensus_calculator
        self.response_synthesizer = response_synthesizer
        self.query_enhancer = query_enhancer
        self.worker_communicator = worker_communicator
        self.logger = logger
        self.foveated_router = foveated_router
        self.enable_query_enhancement = enable_query_enhancement
        self.min_votes = min_votes

    def execute(self, request: QueryRequestDTO) -> QueryResponseDTO:
        """
        Execute query handling pipeline.

        Pipeline:
        1. Query enhancement (SLM)
        2. L1 routing (determine target shards)
        3. Worker communication (gather votes)
        4. L3 consensus (RCP v4)
        5. Response synthesis (collapse superposition)

        Args:
            request: QueryRequestDTO with query and context

        Returns:
            QueryResponseDTO with answer and superposition
        """
        # Log query receipt
        self.logger.log_message(
            trace_id=request.trace_id,
            direction="request",
            message_type="client_query",
            payload={
                "query_text": request.query_text,
                "timestamp_context": request.timestamp_context,
            },
            metadata={"endpoint": "/query"},
        )

        # Step 1: Query Enhancement
        enhanced_query = request.query_text
        if self.enable_query_enhancement:
            enhancement_result = self.query_enhancer.enhance(
                request.query_text,
                request.timestamp_context,
                request.trace_id,
            )
            enhanced_query = enhancement_result.get("enhanced_query", request.query_text)

            self.logger.log_event(
                trace_id=request.trace_id,
                event_type="SYSTEM_INIT",
                data={
                    "original_query": request.query_text,
                    "enhanced_query": enhanced_query,
                    "enhancement_used": enhancement_result.get("enhancement_used", False),
                },
            )

        # Step 2: L1 Routing
        
        # Apply Foveated Routing if enabled to restrict search space
        restrict_ranges = None
        if self.foveated_router:
            restrict_ranges = self.foveated_router.get_semantic_time_ranges(enhanced_query)
            if restrict_ranges:
                self.logger.log_event(
                    trace_id=request.trace_id,
                    event_type="FOVEATED_ROUTING",
                    data={"ranges": restrict_ranges}
                )

        target_shards = self.routing_service.get_target_shards(
            timestamp_context=request.timestamp_context,
            restrict_to_ranges=restrict_ranges
        )

        # Step 3: Gather Votes
        votes = self.worker_communicator.gather_votes(
            trace_id=request.trace_id,
            query_text=enhanced_query,
            original_query=request.query_text,
            target_shards=target_shards,
            timestamp_context=request.timestamp_context,
        )

        # Handle no votes case
        if not votes:
            self.logger.log_event(
                trace_id=request.trace_id,
                event_type="HALLUCINATION_DETECTED",
                data={"reason": "No votes received"},
            )

            return QueryResponseDTO(
                trace_id=request.trace_id,
                final_answer=None,
                confidence=0.0,
                status="FAILED",
                sources=[],
                superposition={
                    "consensus": [],
                    "polar": [],
                    "negative_consensus": [],
                },
            )

        # Convert votes to domain Vote objects if needed
        domain_votes = []
        for v in votes:
            if hasattr(v, "content_hash"):  # Already a Vote-like object
                domain_votes.append(v)
            else:  # Convert from ConsensusVote model
                domain_votes.append(
                    Vote(
                        content_hash=v.content_hash,
                        content_snippet=v.content_snippet,
                        cluster_id=v.cluster_id,
                        binary_vote=v.binary_vote,
                        author_cluster=v.author_cluster,
                        confidence_score=v.confidence_score,
                        document_ids=v.document_ids or [],
                    )
                )

        # Step 4: L3 Consensus Calculation
        consensus_result = self.consensus_calculator.calculate_consensus(domain_votes)

        # Step 5: Response Synthesis
        collapsed_response = self.response_synthesizer.synthesize_response(
            consensus_result
        )

        # Log final response
        self.logger.log_message(
            trace_id=request.trace_id,
            direction="response",
            message_type="final_response",
            payload=collapsed_response.to_dict(),
            metadata={"votes_received": len(votes)},
        )

        return QueryResponseDTO(
            trace_id=request.trace_id,
            final_answer=collapsed_response.final_answer,
            confidence=collapsed_response.confidence,
            status=collapsed_response.status,
            sources=collapsed_response.sources,
            superposition=collapsed_response.superposition.to_dict(),
        )
