"""
SimpleRouterService

Simple implementation of IRouterService using time-based sharding.
This is the current production routing logic from active_agent.py.
"""

from typing import List, Optional
from dpr_rc.application.interfaces import IRouterService


class SimpleRouterService(IRouterService):
    """
    Simple time-based routing service.

    This implements the EXACT same logic as RouteLogic.get_target_shards
    in active_agent.py, ensuring identical routing decisions.

    Per DPR Spec Section 3.1.1: "The Central Index is partitioned into Time-Based Shards"
    """

    def get_target_shards(
        self,
        query_text: str,
        timestamp_context: Optional[str] = None
    ) -> List[str]:
        """
        Determine target shards based on timestamp context.

        Current implementation:
        - If timestamp_context provided: extract year and target that shard
        - Otherwise: broadcast to all shards

        Args:
            query_text: Query text (currently unused, reserved for future intent detection)
            timestamp_context: Optional temporal context (e.g., "2023-01-01")

        Returns:
            List of shard IDs (e.g., ["shard_2023"] or ["broadcast"])
        """
        if timestamp_context:
            # Deterministic sharding logic based on year
            year = timestamp_context[:4]
            return [f"shard_{year}"]
        return ["broadcast"]
