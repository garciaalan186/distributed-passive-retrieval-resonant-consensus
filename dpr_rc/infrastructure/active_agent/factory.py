"""
Infrastructure: Active Agent Factory

Dependency injection factory for assembling all active agent components.
"""

import os
from typing import Optional
import redis

from dpr_rc.application.active_agent import HandleQueryUseCase
from dpr_rc.domain.active_agent.services import (
    RoutingService,
    ConsensusCalculator,
    ResponseSynthesizer,
)
from dpr_rc.infrastructure.active_agent.clients import (
    QueryEnhancerClient,
    WorkerCommunicator,
)
from dpr_rc.infrastructure.active_agent.loaders import ManifestLoader
from dpr_rc.infrastructure.passive_agent.adapters import LoggerAdapter  # Reuse
from dpr_rc.logging_utils import StructuredLogger, ComponentType
from dpr_rc.models import RCPConfig
from dpr_rc.embedding_utils import GCSEmbeddingStore


class ActiveAgentFactory:
    """
    Factory for creating Active Agent components.

    Implements dependency injection pattern.
    """

    @staticmethod
    def create_handle_query_use_case(
        slm_url: str,
        worker_urls: list[str],
        bucket_name: Optional[str],
        redis_client: Optional[redis.Redis] = None,
        scale: str = "medium",
        use_http_workers: bool = True,
        enable_query_enhancement: bool = True,
        min_votes: int = 1,
        rcp_config: Optional[RCPConfig] = None,
    ) -> HandleQueryUseCase:
        """
        Create fully wired HandleQueryUseCase.

        Args:
            slm_url: SLM service URL
            worker_urls: List of worker HTTP endpoints
            bucket_name: GCS bucket for manifests
            redis_client: Optional Redis client
            scale: Data scale (small/medium/large)
            use_http_workers: Use HTTP instead of Redis
            enable_query_enhancement: Enable SLM query enhancement
            min_votes: Minimum votes for consensus
            rcp_config: RCP v4 configuration

        Returns:
            Fully configured HandleQueryUseCase instance
        """
        rcp_config = rcp_config or RCPConfig()

        # Infrastructure: Load manifests
        manifest_loader = ManifestLoader(bucket_name=bucket_name, scale=scale)
        manifest, causal_index = manifest_loader.load_manifests()

        # Helper for shard discovery
        shard_discovery_callback = None
        if bucket_name:
            def discover_shards():
                store = GCSEmbeddingStore(bucket_name)
                return store.list_available_shards(scale)
            shard_discovery_callback = discover_shards

        # Domain Services
        routing_service = RoutingService(
            manifest=manifest, 
            causal_index=causal_index,
            shard_discovery_callback=shard_discovery_callback
        )

        consensus_calculator = ConsensusCalculator(
            theta=rcp_config.theta,
            tau=rcp_config.tau,
        )

        response_synthesizer = ResponseSynthesizer(
            consensus_confidence=0.95,
            polar_confidence=0.70,
            negative_consensus_confidence=0.40,
        )

        # Infrastructure: Clients
        query_enhancer = QueryEnhancerClient(slm_service_url=slm_url, timeout=30.0)

        worker_communicator = WorkerCommunicator(
            worker_urls=worker_urls,
            redis_client=redis_client,
            use_http=use_http_workers,
            http_timeout=90.0,
            redis_timeout=5.0,
        )

        # Infrastructure: Logger
        structured_logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)
        logger = LoggerAdapter(structured_logger)

        # Application: Use Case
        use_case = HandleQueryUseCase(
            routing_service=routing_service,
            consensus_calculator=consensus_calculator,
            response_synthesizer=response_synthesizer,
            query_enhancer=query_enhancer,
            worker_communicator=worker_communicator,
            logger=logger,
            enable_query_enhancement=enable_query_enhancement,
            min_votes=min_votes,
        )

        return use_case

    @staticmethod
    def create_from_env() -> HandleQueryUseCase:
        """
        Create use case from environment variables.

        Convenience method for production deployment.

        Returns:
            Configured HandleQueryUseCase
        """
        # Read environment variables
        slm_url = os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
        passive_worker_url = os.getenv("PASSIVE_WORKER_URL", "")
        worker_urls = [url.strip() for url in passive_worker_url.split(",") if url.strip()]

        bucket_name = os.getenv("HISTORY_BUCKET")
        scale = os.getenv("HISTORY_SCALE", "medium")
        use_http_workers = os.getenv("USE_HTTP_WORKERS", "true").lower() == "true"
        enable_query_enhancement = (
            os.getenv("ENABLE_QUERY_ENHANCEMENT", "true").lower() == "true"
        )
        min_votes = int(os.getenv("MIN_VOTES", "1"))

        # Optional Redis
        redis_host = os.getenv("REDIS_HOST")
        redis_client = None
        if redis_host:
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            try:
                redis_client = redis.Redis(
                    host=redis_host, port=redis_port, decode_responses=True
                )
                redis_client.ping()  # Test connection
            except Exception:
                redis_client = None

        # RCP configuration
        rcp_config = RCPConfig()

        return ActiveAgentFactory.create_handle_query_use_case(
            slm_url=slm_url,
            worker_urls=worker_urls,
            bucket_name=bucket_name,
            redis_client=redis_client,
            scale=scale,
            use_http_workers=use_http_workers,
            enable_query_enhancement=enable_query_enhancement,
            min_votes=min_votes,
            rcp_config=rcp_config,
        )
