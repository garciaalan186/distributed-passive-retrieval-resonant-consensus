"""
SimpleRouterService

Tempo-normalized routing implementation using local dataset shard discovery.
"""

import os
import json
from typing import List, Optional
from dpr_rc.application.interfaces import IRouterService
from dpr_rc.domain.active_agent.services.routing_service import RoutingService


class SimpleRouterService(IRouterService):
    """
    Tempo-normalized routing service with local shard discovery.

    Discovers available shards from local dataset and routes queries based on
    tempo-normalized shard date ranges.

    Per DPR Spec Section 3.1.1: "The Central Index is partitioned into Time-Based Shards"
    """

    def __init__(self):
        """Initialize router with local-based shard discovery."""
        self._routing_service = RoutingService(
            shard_discovery_callback=self._discover_shards_from_local
        )

    def get_target_shards(
        self,
        query_text: str,
        timestamp_context: Optional[str] = None
    ) -> List[str]:
        """
        Determine target shards based on timestamp context.

        Uses tempo-normalized routing:
        1. Discovers available shards from local dataset
        2. Parses shard date ranges
        3. Returns shards containing the timestamp

        Args:
            query_text: Query text (currently unused)
            timestamp_context: Optional temporal context (e.g., "2015-12-31")

        Returns:
            List of shard IDs (e.g., ["shard_2015"])
        """
        return self._routing_service.get_target_shards(timestamp_context)

    def _discover_shards_from_local(self) -> List[str]:
        """
        Discover available shards from local dataset.

        Returns:
            List of shard filenames (e.g., ["shard_2015.json"])
        """
        local_path = os.getenv("LOCAL_DATASET_PATH")
        if not local_path or not os.path.exists(local_path):
            return []

        try:
            with open(local_path, 'r') as f:
                data = json.load(f)

            # Extract unique years from events
            events = data.get("events", [])
            years = set()
            for event in events:
                ts = event.get("timestamp", "")
                if ts and len(ts) >= 4:
                    years.add(ts[:4])

            return [f"shard_{year}.json" for year in sorted(years)]

        except Exception as e:
            print(f"Failed to parse local dataset for shards: {e}")
            return []
