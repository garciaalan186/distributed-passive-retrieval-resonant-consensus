"""
Domain Entities for Passive Agent

These are pure domain objects with no external dependencies.
They represent core business concepts in the DPR-RC system.
"""

from .verification_result import VerificationResult
from .quadrant import QuadrantCoordinates
from .shard import ShardMetadata, LoadStrategy

__all__ = [
    "VerificationResult",
    "QuadrantCoordinates",
    "ShardMetadata",
    "LoadStrategy",
]
