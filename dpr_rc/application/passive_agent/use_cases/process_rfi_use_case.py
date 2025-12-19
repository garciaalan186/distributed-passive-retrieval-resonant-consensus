"""
Use Case: Process RFI

Application layer orchestrator for RFI processing.
Coordinates domain services and repositories to handle requests.
"""

from typing import Dict, Any, Optional, Protocol
from ..dtos import ProcessRFIRequest, ProcessRFIResponse
from dpr_rc.domain.passive_agent.services import RFIProcessor
from dpr_rc.domain.passive_agent.repositories import IShardRepository


class ILogger(Protocol):
    """Interface for logging operations."""

    def log_message(
        self,
        trace_id: str,
        direction: str,
        message_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a message with structured metadata."""
        ...

    def log_event(
        self,
        trace_id: str,
        event_type: str,
        data: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an event with metrics."""
        ...


class ProcessRFIUseCase:
    """
    Use case for processing RFI requests.

    Orchestrates:
    1. Shard loading (lazy)
    2. RFI processing via domain service
    3. Logging and audit trail
    """

    def __init__(
        self,
        shard_repository: IShardRepository,
        rfi_processor: RFIProcessor,
        logger: ILogger,
        worker_id: str,
        cluster_id: str,
        default_epoch_year: int = 2020,
    ):
        """
        Initialize use case.

        Args:
            shard_repository: Repository for loading shards
            rfi_processor: Domain service for RFI processing
            logger: Logger for audit trail
            worker_id: Worker identifier
            cluster_id: Cluster identifier
            default_epoch_year: Default epoch if no target shards specified
        """
        self.shard_repository = shard_repository
        self.rfi_processor = rfi_processor
        self.logger = logger
        self.worker_id = worker_id
        self.cluster_id = cluster_id
        self.default_epoch_year = default_epoch_year

    def execute(self, request: ProcessRFIRequest) -> ProcessRFIResponse:
        """
        Execute RFI processing.

        Args:
            request: ProcessRFIRequest with query and target shards

        Returns:
            ProcessRFIResponse with votes or error status
        """
        # Log RFI receipt
        self.logger.log_message(
            trace_id=request.trace_id,
            direction="request",
            message_type="rfi_received",
            payload={
                "query_text": request.query_text,
                "original_query": request.original_query,
                "target_shards": request.target_shards,
            },
            metadata={"worker_id": self.worker_id, "cluster": self.cluster_id},
        )

        # Handle default shards
        target_shards = request.target_shards
        if not target_shards or "broadcast" in target_shards:
            target_shards = [f"shard_{self.default_epoch_year}"]

        # Lazy load shards
        try:
            for shard_id in target_shards:
                if not self.shard_repository.is_shard_loaded(shard_id):
                    self.shard_repository.load_shard(shard_id)
        except Exception as e:
            return ProcessRFIResponse(
                trace_id=request.trace_id,
                votes=[],
                worker_id=self.worker_id,
                cluster_id=self.cluster_id,
                status="error",
                message=f"Failed to load shards: {e}",
            )

        # Process RFI through domain service
        try:
            votes = self.rfi_processor.process_rfi(
                trace_id=request.trace_id,
                query_text=request.query_text,
                original_query=request.original_query,
                target_shards=target_shards,
                foveated_context=None,  # TODO: Load from shard metadata
            )

            # Convert votes to dict format
            vote_dicts = [vote.to_dict() for vote in votes]

            # Log votes
            for vote in votes:
                self.logger.log_message(
                    trace_id=request.trace_id,
                    direction="internal",
                    message_type="vote_created",
                    payload=vote.to_dict(),
                    metadata={
                        "worker_id": self.worker_id,
                        "cluster": self.cluster_id,
                        "shard_id": target_shards,
                    },
                )

                self.logger.log_event(
                    trace_id=request.trace_id,
                    event_type="VOTE_CAST",
                    data=vote.to_dict(),
                    metrics={
                        "confidence": vote.confidence_score,
                        "shard": target_shards[0] if target_shards else "unknown",
                    },
                )

            # Determine status
            if vote_dicts:
                status = "success"
                message = f"Processed {len(vote_dicts)} vote(s)"
            else:
                status = "no_results"
                message = "No relevant content found"

            return ProcessRFIResponse(
                trace_id=request.trace_id,
                votes=vote_dicts,
                worker_id=self.worker_id,
                cluster_id=self.cluster_id,
                status=status,
                message=message,
            )

        except Exception as e:
            return ProcessRFIResponse(
                trace_id=request.trace_id,
                votes=[],
                worker_id=self.worker_id,
                cluster_id=self.cluster_id,
                status="error",
                message=f"Processing failed: {e}",
            )
