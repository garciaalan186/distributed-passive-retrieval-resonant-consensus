"""
Domain Entity: Shard

Represents a temporal shard of historical data.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class LoadStrategy(str, Enum):
    """Strategy used to load shard data."""

    GCS_EMBEDDINGS = "gcs_embeddings"  # Pre-computed embeddings from GCS
    GCS_RAW = "gcs_raw"  # Raw JSON data from GCS (compute embeddings)
    REDIS_CACHE = "redis_cache"  # Cached data from Redis
    FALLBACK = "fallback"  # Generated fallback data


@dataclass
class ShardMetadata:
    """
    Metadata about a temporal shard.

    Shards represent frozen snapshots of history at specific time periods.
    """

    shard_id: str  # e.g., "shard_2020"
    year: int  # Temporal year this shard represents
    document_count: int  # Number of documents in the shard
    embedding_model: Optional[str] = None  # Model used for embeddings
    loaded_from: LoadStrategy = LoadStrategy.FALLBACK  # How was it loaded

    @property
    def is_real_data(self) -> bool:
        """Check if shard contains real data (not fallback)."""
        return self.loaded_from in (
            LoadStrategy.GCS_EMBEDDINGS,
            LoadStrategy.GCS_RAW,
            LoadStrategy.REDIS_CACHE,
        )

    @classmethod
    def from_shard_id(cls, shard_id: str) -> "ShardMetadata":
        """
        Create metadata from shard_id by parsing the year.

        Args:
            shard_id: Shard identifier like "shard_2020"

        Returns:
            ShardMetadata with year extracted
        """
        try:
            year_str = shard_id.replace("shard_", "")
            year = int(year_str)
        except (ValueError, AttributeError):
            year = 2020  # Default fallback year

        return cls(shard_id=shard_id, year=year, document_count=0)
