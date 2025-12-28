"""
Repository Interface: Shard Repository

Defines the contract for loading and managing temporal shards.
"""

from typing import Protocol, Optional, Dict, Any
from ..entities import ShardMetadata


class IShardRepository(Protocol):
    """
    Interface for shard data access.

    This repository handles loading shards from various sources
    (GCS embeddings, GCS raw, Redis cache, or fallback generation).
    """

    def load_shard(self, shard_id: str) -> ShardMetadata:
        """
        Load a shard by ID using the lazy loading strategy.

        The repository tries multiple strategies in order:
        1. Load pre-computed embeddings from GCS
        2. Load raw JSON from GCS and compute embeddings
        3. Load from Redis cache
        4. Generate fallback data

        Args:
            shard_id: Identifier like "shard_2020"

        Returns:
            ShardMetadata describing the loaded shard

        Raises:
            Exception if all loading strategies fail
        """
        ...

    def get_shard_data(self, shard_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the raw data for a loaded shard.

        Args:
            shard_id: Shard identifier

        Returns:
            Dictionary containing shard data, or None if not loaded
        """
        ...

    def is_shard_loaded(self, shard_id: str) -> bool:
        """
        Check if a shard is already loaded in memory.

        Args:
            shard_id: Shard identifier

        Returns:
            True if shard is loaded, False otherwise
        """
        ...
