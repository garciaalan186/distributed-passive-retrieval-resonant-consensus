"""
Repository Interfaces for Passive Agent

These interfaces define contracts for data access without
specifying implementation details (Dependency Inversion Principle).
"""

from .shard_repository import IShardRepository
from .embedding_repository import IEmbeddingRepository, RetrievalResult

__all__ = [
    "IShardRepository",
    "IEmbeddingRepository",
    "RetrievalResult",
]
