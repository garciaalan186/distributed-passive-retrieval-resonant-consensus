"""
Domain Service: Routing Service

L1 time-sharded routing logic with tempo-normalized sharding and causal awareness.
"""

from typing import List, Dict, Optional


class RoutingService:
    """
    Domain service for L1 routing decisions.

    Determines target shards based on temporal context.
    """

    def __init__(
        self,
        manifest: Optional[Dict] = None,
        causal_index: Optional[Dict] = None,
    ):
        """
        Initialize routing service.

        Args:
            manifest: Shard manifest with time ranges
            causal_index: Causal dependency index
        """
        self.manifest = manifest
        self.causal_index = causal_index

    def get_target_shards(
        self, timestamp_context: Optional[str] = None
    ) -> List[str]:
        """
        Determine target shards based on timestamp context.

        Enhanced routing:
        1. Query manifest to find primary shard(s) for timestamp
        2. Expand to include causal ancestor shards (up to depth 2)
        3. Fallback to legacy year-based routing if indices unavailable

        Args:
            timestamp_context: ISO timestamp or None

        Returns:
            List of shard IDs to query
        """
        if not timestamp_context:
            return ["broadcast"]

        # Use tempo-normalized routing if manifest available
        if self.manifest and "shards" in self.manifest:
            return self._get_tempo_normalized_shards(timestamp_context)

        # Fallback to legacy year-based routing
        year = timestamp_context[:4]
        return [f"shard_{year}"]

    def _get_tempo_normalized_shards(self, timestamp: str) -> List[str]:
        """
        Get tempo-normalized shards for timestamp with causal expansion.

        Args:
            timestamp: ISO timestamp

        Returns:
            List of shard IDs (primary + causal ancestors)
        """
        primary_shards = []

        # Find shards containing this timestamp
        for shard in self.manifest.get("shards", []):
            time_range = shard.get("time_range", {})
            start = time_range.get("start", "")
            end = time_range.get("end", "")

            if start <= timestamp <= end:
                primary_shards.append(shard.get("id", ""))

        if not primary_shards:
            # No shard found, fallback
            return ["broadcast"]

        # Expand to include causal ancestors
        if self.causal_index and "shard_ancestry" in self.causal_index:
            expanded_shards = set(primary_shards)

            for shard_id in primary_shards:
                ancestry = self.causal_index["shard_ancestry"].get(shard_id, {})
                # Include direct ancestors (depth 1)
                direct_ancestors = ancestry.get("direct_ancestors", [])
                expanded_shards.update(direct_ancestors[:3])  # Limit to top 3

            return sorted(list(expanded_shards))

        return primary_shards
