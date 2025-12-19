"""
Domain Entity: QuadrantCoordinates

Represents semantic quadrant coordinates in the RCP v4 topological space.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class QuadrantCoordinates:
    """
    Semantic quadrant coordinates <v+, v->.

    Per RCP v4 specification:
    - v+ : approval rate from cluster that created the artifact
    - v- : approval rate from other clusters

    These coordinates map responses to a 2D topological space,
    enabling classification into consensus/polar/negative_consensus regions.
    """

    v_plus: float  # Approval from author's cluster
    v_minus: float  # Approval from other clusters

    def to_list(self) -> List[float]:
        """Convert to list format for serialization."""
        return [self.v_plus, self.v_minus]

    @classmethod
    def from_list(cls, coords: List[float]) -> "QuadrantCoordinates":
        """Create from list format [v+, v-]."""
        if len(coords) != 2:
            raise ValueError(f"Expected 2 coordinates, got {len(coords)}")
        return cls(v_plus=coords[0], v_minus=coords[1])

    @classmethod
    def calculate(
        cls,
        author_cluster: str,
        worker_cluster: str,
        cluster_approvals: dict[str, float],
    ) -> "QuadrantCoordinates":
        """
        Calculate quadrant coordinates from cluster approval rates.

        Args:
            author_cluster: Cluster that authored the artifact
            worker_cluster: Current worker's cluster
            cluster_approvals: Dict mapping cluster_id -> approval_rate

        Returns:
            QuadrantCoordinates with v+ and v- calculated
        """
        # v+ = approval from author's cluster
        v_plus = cluster_approvals.get(author_cluster, 0.0)

        # v- = average approval from all other clusters
        other_approvals = [
            rate
            for cluster_id, rate in cluster_approvals.items()
            if cluster_id != author_cluster
        ]
        v_minus = sum(other_approvals) / len(other_approvals) if other_approvals else 0.0

        return cls(v_plus=v_plus, v_minus=v_minus)
