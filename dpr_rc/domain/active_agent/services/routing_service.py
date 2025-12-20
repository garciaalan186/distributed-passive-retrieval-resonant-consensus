"""
Domain Service: Routing Service

L1 time-sharded routing logic with tempo-normalized sharding and causal awareness.
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Type alias for TimeRange
TimeRange = Tuple[str, str]


@dataclass
class ShardInfo:
    """Information about a tempo-normalized shard."""
    shard_id: str
    start_date: str  # YYYY-MM format
    end_date: str    # YYYY-MM format


class RoutingService:
    """
    Domain service for L1 routing decisions.

    Determines target shards based on temporal context using tempo-normalized
    shard naming convention: shard_{id}_{start_YYYY-MM}_{end_YYYY-MM}
    """

    def __init__(
        self,
        manifest: Optional[Dict] = None,
        causal_index: Optional[Dict] = None,
        shard_discovery_callback: Optional[callable] = None,
    ):
        """
        Initialize routing service.

        Args:
            manifest: Shard manifest with time ranges (optional, for future use)
            causal_index: Causal dependency index (optional)
            shard_discovery_callback: Function to discover available shards from GCS
        """
        self.manifest = manifest
        self.causal_index = causal_index
        self.shard_discovery_callback = shard_discovery_callback
        self._shard_cache: Optional[List[ShardInfo]] = None

    def get_target_shards(
        self, 
        timestamp_context: Optional[str] = None,
        restrict_to_ranges: Optional[List[TimeRange]] = None
    ) -> List[str]:
        """
        Determine target shards based on timestamp context.

        Routing strategy:
        1. Parse tempo-normalized shard names from available shards
        2. Find shards whose date range contains the timestamp
        3. Return matching shard IDs (without file extension)

        Args:
            timestamp_context: ISO timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
            restrict_to_ranges: Optional list of (start, end) tuples to restrict search.
                                If provided, only shards overlapping these ranges AND matches
                                the timestamp (if present) are returned.

        Returns:
            List of shard IDs to query (e.g., ["shard_000_2015-01_2021-12"])
        """
        if not timestamp_context:
            return ["broadcast"]

        # Use tempo-normalized routing with dynamic shard discovery
        return self._get_tempo_normalized_shards(timestamp_context, restrict_to_ranges)

    def _get_tempo_normalized_shards(
        self, 
        timestamp: Optional[str],
        restrict_to_ranges: Optional[List[TimeRange]] = None
    ) -> List[str]:
        """
        Get tempo-normalized shards for timestamp.

        Discovers available shards and matches them to the timestamp.

        Args:
            timestamp: ISO timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)

        Returns:
            List of shard IDs matching the timestamp
        """
        # Discover available shards
        available_shards = self._discover_shards()

        if not available_shards:
            # No shards discovered, fallback to broadcast
            return ["broadcast"]

        # Extract YYYY-MM from timestamp for comparison if present
        query_date = timestamp[:7] if timestamp else None

        # Find shards containing this timestamp
        matching_shards = []
        for shard in available_shards:
            # Check specific timestamp match if provided
            timestamp_match = True
            if query_date:
                timestamp_match = (shard.start_date <= query_date <= shard.end_date)
            
            # Check range restriction if provided
            range_match = True
            if restrict_to_ranges:
                range_match = False
                for r_start, r_end in restrict_to_ranges:
                    # Check overlap: shard_start <= range_end AND shard_end >= range_start
                    if shard.start_date <= r_end and shard.end_date >= r_start:
                        range_match = True
                        break

            if timestamp_match and range_match:
                matching_shards.append(shard.shard_id)

        if not matching_shards:
            # No exact match, return broadcast
            return ["broadcast"]

        return sorted(matching_shards)

    def _discover_shards(self) -> List[ShardInfo]:
        """
        Discover available tempo-normalized shards.

        Uses callback if provided, otherwise returns cached results.

        Returns:
            List of ShardInfo objects
        """
        # Return cached shards if available
        if self._shard_cache is not None:
            return self._shard_cache

        # Use discovery callback if provided
        if self.shard_discovery_callback:
            shard_names = self.shard_discovery_callback()
            self._shard_cache = self._parse_shard_names(shard_names)
            return self._shard_cache

        # No discovery mechanism available
        return []

    def _parse_shard_names(self, shard_names: List[str]) -> List[ShardInfo]:
        """
        Parse tempo-normalized shard names.

        Expected format: shard_{id}_{start_YYYY-MM}_{end_YYYY-MM}.{ext}
        Example: shard_000_2015-01_2021-12.json

        Args:
            shard_names: List of shard filenames or paths

        Returns:
            List of ShardInfo objects
        """
        shard_pattern = re.compile(
            r'shard_(\d+)_(\d{4}-\d{2})_(\d{4}-\d{2})'
        )

        shards = []
        for name in shard_names:
            # Extract filename from path if necessary
            filename = os.path.basename(name)

            match = shard_pattern.search(filename)
            if match:
                shard_id_num, start_date, end_date = match.groups()
                # Construct shard_id without extension
                shard_id = f"shard_{shard_id_num}_{start_date}_{end_date}"
                shards.append(ShardInfo(
                    shard_id=shard_id,
                    start_date=start_date,
                    end_date=end_date
                ))

        return shards
