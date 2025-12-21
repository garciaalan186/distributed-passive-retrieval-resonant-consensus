"""
Domain Entities for Active Agent
"""

from .consensus_state import (
    ConsensusTier,
    ArtifactConsensus,
    ConsensusResult,
)
from .resonance_matrix import ResonanceMatrix, CollapsedResponse

__all__ = [
    "ConsensusTier",
    "ArtifactConsensus",
    "ConsensusResult",
    "ResonanceMatrix",
    "CollapsedResponse",
]
