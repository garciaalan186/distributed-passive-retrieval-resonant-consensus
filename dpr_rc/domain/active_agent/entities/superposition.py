"""
Domain Entity: Superposition

Represents the superposition state - all possible answers classified by consensus tier.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from .consensus_state import ArtifactConsensus, ConsensusResult


@dataclass
class SuperpositionState:
    """
    Quantum-inspired superposition of possible answers.

    Per RCP v4: Artifacts are classified into tiers before collapse.
    """
    consensus: List[Dict]  # High-confidence answers
    polar: List[Dict]  # Contested answers
    negative_consensus: List[Dict]  # Cross-cluster rejected answers

    @classmethod
    def from_consensus_result(cls, result: ConsensusResult) -> "SuperpositionState":
        """Create superposition from consensus result."""
        return cls(
            consensus=[a.to_dict() for a in result.consensus_artifacts],
            polar=[a.to_dict() for a in result.polar_artifacts],
            negative_consensus=[a.to_dict() for a in result.negative_consensus_artifacts],
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "consensus": self.consensus,
            "polar": self.polar,
            "negative_consensus": self.negative_consensus,
        }

    def is_empty(self) -> bool:
        """Check if superposition has no states."""
        return (
            len(self.consensus) == 0
            and len(self.polar) == 0
            and len(self.negative_consensus) == 0
        )


@dataclass
class CollapsedResponse:
    """
    Result after collapsing superposition to a single answer.

    Represents the final output to the user.
    """
    final_answer: Optional[str]
    confidence: Optional[float]
    status: str  # "SUCCESS", "FAILED", "NO_CONSENSUS"
    sources: List[str]
    superposition: SuperpositionState

    @classmethod
    def from_best_artifact(
        cls, artifact: ArtifactConsensus, superposition: SuperpositionState
    ) -> "CollapsedResponse":
        """Create response from best artifact."""
        # Map tier to confidence
        confidence_map = {
            "CONSENSUS": 0.95,
            "POLAR": 0.70,
            "NEGATIVE_CONSENSUS": 0.40,
        }

        # Use source document IDs instead of content hash
        sources = artifact.source_document_ids if hasattr(artifact, 'source_document_ids') and artifact.source_document_ids else []

        return cls(
            final_answer=artifact.content,
            confidence=confidence_map.get(artifact.tier.value, 0.0),
            status="SUCCESS",
            sources=sources,
            superposition=superposition,
        )

    @classmethod
    def no_consensus(cls, superposition: SuperpositionState) -> "CollapsedResponse":
        """Create response when no consensus reached."""
        return cls(
            final_answer=None,
            confidence=0.0,
            status="NO_CONSENSUS",
            sources=[],
            superposition=superposition,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "status": self.status,
            "sources": self.sources,
            "superposition": self.superposition.to_dict(),
        }
