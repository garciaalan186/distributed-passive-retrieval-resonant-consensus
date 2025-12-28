"""
SimpleRouterService

Tempo-normalized routing implementation using GCS shard discovery.
"""

import os
from typing import List, Optional
from dpr_rc.application.interfaces import IRouterService
from dpr_rc.domain.active_agent.services.routing_service import RoutingService


class SimpleRouterService(IRouterService):
    """
    Tempo-normalized routing service with GCS shard discovery.

    Discovers available shards from GCS bucket and routes queries based on
    tempo-normalized shard date ranges.

    Per DPR Spec Section 3.1.1: "The Central Index is partitioned into Time-Based Shards"
    """

    def __init__(self):
        """Initialize router with GCS-based shard discovery."""
        self._routing_service = RoutingService(
            shard_discovery_callback=self._discover_shards_from_gcs
        )

    def get_target_shards(
        self,
        query_text: str,
        timestamp_context: Optional[str] = None
    ) -> List[str]:
        """
        Determine target shards based on timestamp context.

        Uses tempo-normalized routing:
        1. Discovers available shards from GCS
        2. Parses shard date ranges (shard_{id}_{start}_{end})
        3. Returns shards containing the timestamp

        Args:
            query_text: Query text (currently unused)
            timestamp_context: Optional temporal context (e.g., "2015-12-31")

        Returns:
            List of shard IDs (e.g., ["shard_000_2015-01_2021-12"])
        """
        return self._routing_service.get_target_shards(timestamp_context)

    def _discover_shards_from_gcs(self) -> List[str]:
        """
        Discover available shards from GCS bucket.

        Returns:
            List of shard filenames (e.g., ["shard_000_2015-01_2021-12.json"])
        """
        try:
            from google.cloud import storage
        except ImportError:
            # google-cloud-storage not available, return empty
            return []

        bucket_name = os.getenv("HISTORY_BUCKET", "")
        scale = os.getenv("HISTORY_SCALE", "small")

        if not bucket_name:
            return []

        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # List shards from raw data directory
            prefix = f"raw/{scale}/shards/"
            blobs = bucket.list_blobs(prefix=prefix)

            shard_names = []
            for blob in blobs:
                if blob.name.endswith('.json'):
                    shard_names.append(blob.name)

            return shard_names

        except Exception:
            # GCS error, return empty (will fallback to broadcast)
            return []
