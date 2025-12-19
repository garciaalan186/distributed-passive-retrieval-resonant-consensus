"""
IRouterService Interface

Interface for L1 routing logic that determines target shards.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class IRouterService(ABC):
    """
    Interface for L1 Time-Sharded Routing Logic.

    Per DPR Architecture Spec Section 3.1.1:
    "The Central Index is partitioned into Time-Based Shards"

    This interface enables dependency inversion and makes routing
    logic testable and swappable.
    """

    @abstractmethod
    def get_target_shards(
        self,
        query_text: str,
        timestamp_context: Optional[str] = None
    ) -> List[str]:
        """
        Determine target shards based on query and temporal context.

        Args:
            query_text: The query text (may be used for intent detection)
            timestamp_context: Optional temporal context (e.g., "2023-01-01")

        Returns:
            List of shard IDs to query (e.g., ["shard_2023"] or ["broadcast"])

        Note:
            Current implementation uses simple year-based sharding.
            Future implementations may use more sophisticated routing
            based on query intent, entity extraction, etc.
        """
        pass
