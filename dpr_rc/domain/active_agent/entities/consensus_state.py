"""
Domain Entity: Consensus State

Represents the state of consensus calculation for a query.
"""

from dataclasses import dataclass
from typing import List, Dict, Set
from enum import Enum


class ConsensusTier(str, Enum):
    """RCP v4 classification tiers."""
    CONSENSUS = "CONSENSUS"
    POLAR = "POLAR"
    NEGATIVE_CONSENSUS = "NEGATIVE_CONSENSUS"
    REJECT = "REJECT"


@dataclass
class ArtifactConsensus:
    """
    Consensus state for a single artifact.

    Represents the result of applying RCP v4 equations to one candidate answer.
    """
    content_hash: str
    content: str
    votes_count: int
    approval_set: Set[str]  # Cluster IDs that approve
    agreement_ratio: float  # ρ(ω) per RCP v4 Eq. 3
    tier: ConsensusTier  # Classification per RCP v4 Eq. 4
    score: float  # Artifact score per RCP v4 Eq. 5
    resonance_vector: List[float]  # [v+, v-] coordinates
    source_document_ids: List[str] = None  # Source document IDs for provenance

    def __post_init__(self):
        """Ensure source_document_ids is initialized."""
        if self.source_document_ids is None:
            self.source_document_ids = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "content_hash": self.content_hash,
            "content": self.content,
            "votes_count": self.votes_count,
            "approval_set": list(self.approval_set),
            "agreement_ratio": self.agreement_ratio,
            "tier": self.tier.value,
            "score": self.score,
            "resonance_vector": self.resonance_vector,
            "source_document_ids": self.source_document_ids,
        }


@dataclass
class ConsensusResult:
    """
    Overall consensus result for a query.

    Contains all artifacts classified by tier.
    """
    consensus_artifacts: List[ArtifactConsensus]
    polar_artifacts: List[ArtifactConsensus]
    negative_consensus_artifacts: List[ArtifactConsensus]
    total_votes: int
    num_clusters: int

    def has_consensus(self) -> bool:
        """Check if any artifact reached consensus tier."""
        return len(self.consensus_artifacts) > 0

    def has_any_results(self) -> bool:
        """Check if any artifacts were found."""
        return (
            len(self.consensus_artifacts) > 0
            or len(self.polar_artifacts) > 0
            or len(self.negative_consensus_artifacts) > 0
        )

    def get_best_artifact(self) -> ArtifactConsensus:
        """
        Get the best artifact based on tier priority.

        Priority: Consensus > Polar > Negative Consensus
        Within tier: highest score wins
        """
        if self.consensus_artifacts:
            return max(self.consensus_artifacts, key=lambda a: a.score)
        elif self.polar_artifacts:
            return max(self.polar_artifacts, key=lambda a: a.score)
        elif self.negative_consensus_artifacts:
            return max(self.negative_consensus_artifacts, key=lambda a: a.score)
        else:
            raise ValueError("No artifacts available")
