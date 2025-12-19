"""
Infrastructure: Passive Agent Factory

Dependency injection factory for assembling all components.
Single source of truth for component wiring.
"""

import os
from typing import Optional
import redis

from dpr_rc.application.passive_agent import ProcessRFIUseCase
from dpr_rc.domain.passive_agent.services import (
    VerificationService,
    QuadrantService,
    RFIProcessor,
)
from dpr_rc.infrastructure.passive_agent.repositories import (
    GCSShardRepository,
    ChromaDBRepository,
)
from dpr_rc.infrastructure.passive_agent.clients import HttpSLMClient
from dpr_rc.infrastructure.passive_agent.adapters import LoggerAdapter
from dpr_rc.logging_utils import StructuredLogger, ComponentType
from dpr_rc.embedding_utils import DEFAULT_EMBEDDING_MODEL


class PassiveAgentFactory:
    """
    Factory for creating PassiveAgent components.

    Implements dependency injection pattern.
    """

    @staticmethod
    def create_process_rfi_use_case(
        bucket_name: Optional[str],
        slm_url: str,
        worker_id: str,
        cluster_id: str,
        scale: str = "medium",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        redis_client: Optional[redis.Redis] = None,
        vote_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        default_epoch_year: int = 2020,
    ) -> ProcessRFIUseCase:
        """
        Create fully wired ProcessRFIUseCase.

        Args:
            bucket_name: GCS bucket for data (None for local mode)
            slm_url: SLM service URL
            worker_id: Worker identifier
            cluster_id: Cluster identifier
            scale: Data scale (small/medium/large)
            embedding_model: Model for embeddings
            redis_client: Optional Redis client for caching
            vote_threshold: Threshold for binary vote
            confidence_threshold: Minimum confidence to vote
            default_epoch_year: Default epoch year

        Returns:
            Fully configured ProcessRFIUseCase instance
        """
        # Infrastructure: Repositories
        chroma_repo = ChromaDBRepository(
            chroma_client=None,  # Will create default client
            embedding_model=embedding_model,
        )

        shard_repo = GCSShardRepository(
            embedding_repository=chroma_repo,
            bucket_name=bucket_name,
            scale=scale,
            embedding_model=embedding_model,
            redis_client=redis_client,
        )

        # Infrastructure: Clients
        slm_client = HttpSLMClient(
            slm_service_url=slm_url,
            timeout=30,
            worker_id=worker_id,
        )

        # Infrastructure: Logger
        structured_logger = StructuredLogger(ComponentType.PASSIVE_WORKER)
        logger = LoggerAdapter(structured_logger)

        # Domain Services
        verification_service = VerificationService(
            slm_client=slm_client,
            max_retries=3,
            base_delay=2.0,
        )

        quadrant_service = QuadrantService()

        rfi_processor = RFIProcessor(
            embedding_repo=chroma_repo,
            verification_service=verification_service,
            quadrant_service=quadrant_service,
            worker_id=worker_id,
            cluster_id=cluster_id,
            vote_threshold=vote_threshold,
            confidence_threshold=confidence_threshold,
        )

        # Application: Use Case
        use_case = ProcessRFIUseCase(
            shard_repository=shard_repo,
            rfi_processor=rfi_processor,
            logger=logger,
            worker_id=worker_id,
            cluster_id=cluster_id,
            default_epoch_year=default_epoch_year,
        )

        return use_case

    @staticmethod
    def create_from_env() -> ProcessRFIUseCase:
        """
        Create use case from environment variables.

        Convenience method for production deployment.

        Returns:
            Configured ProcessRFIUseCase
        """
        # Read environment variables
        bucket_name = os.getenv("HISTORY_BUCKET")
        slm_url = os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
        worker_id = os.getenv("WORKER_ID", "worker-1")
        cluster_id = os.getenv("CLUSTER_ID", "cluster-alpha")
        scale = os.getenv("HISTORY_SCALE", "medium")
        embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        vote_threshold = float(os.getenv("VOTE_THRESHOLD", "0.5"))
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
        epoch_year = int(os.getenv("EPOCH_YEAR", "2020"))

        # Optional Redis
        redis_host = os.getenv("REDIS_HOST")
        redis_client = None
        if redis_host:
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            try:
                redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True,
                )
            except Exception:
                redis_client = None

        return PassiveAgentFactory.create_process_rfi_use_case(
            bucket_name=bucket_name,
            slm_url=slm_url,
            worker_id=worker_id,
            cluster_id=cluster_id,
            scale=scale,
            embedding_model=embedding_model,
            redis_client=redis_client,
            vote_threshold=vote_threshold,
            confidence_threshold=confidence_threshold,
            default_epoch_year=epoch_year,
        )
