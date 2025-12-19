"""
Domain Service: Quadrant Service

Handles L3 semantic quadrant topology calculation.
Maps responses to topological coordinates <v+, v-> for consensus.
"""

import hashlib
from ..entities import QuadrantCoordinates


class QuadrantService:
    """
    Domain service for L3 semantic quadrant calculation.

    Uses deterministic hashing to compute quadrant coordinates
    based on content and confidence.
    """

    def calculate(self, content: str, confidence: float) -> QuadrantCoordinates:
        """
        Calculate semantic quadrant coordinates from content and confidence.

        Uses deterministic MD5 hash (not Python's hash()) for reproducibility.

        Args:
            content: Response content
            confidence: Verification confidence score

        Returns:
            QuadrantCoordinates with v+ and v- values in [0, 1]
        """
        # Use deterministic hash (MD5) instead of Python's hash()
        # Python's hash() is non-deterministic (randomized seed per process)
        content_hash = int(hashlib.md5(content.encode('utf-8')).hexdigest()[:16], 16)

        # Calculate x coordinate (v+)
        x_base = ((content_hash % 100) / 100.0)
        v_plus = (x_base * 0.5) + (confidence * 0.5)

        # Calculate y coordinate (v-)
        y_base = (((content_hash >> 8) % 100) / 100.0)
        v_minus = (y_base * 0.3) + (confidence * 0.7)

        return QuadrantCoordinates(
            v_plus=round(v_plus, 2),
            v_minus=round(v_minus, 2)
        )

    def compute_binary_vote(self, confidence: float, vote_threshold: float = 0.5) -> int:
        """
        RCP v4 Equation 1: Convert continuous confidence to binary vote.

        v(ω, a) ∈ {0, 1}
        v(ω, a) = 1 if confidence >= threshold, else 0

        Args:
            confidence: Continuous confidence score [0, 1]
            vote_threshold: Threshold for binary decision

        Returns:
            1 if confidence >= threshold, else 0
        """
        return 1 if confidence >= vote_threshold else 0
