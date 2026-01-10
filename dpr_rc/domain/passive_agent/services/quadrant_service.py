"""
Domain Service: Quadrant Service

Handles L3 semantic quadrant topology calculation.
Maps responses to topological coordinates <v+, v-> for consensus.
"""

import hashlib
from ..entities import QuadrantCoordinates
from dpr_rc.config import get_dpr_config

# Load quadrant config
_quadrant_config = get_dpr_config().get('quadrant', {})
_consensus_config = get_dpr_config().get('consensus', {})


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
        # Load config values
        coord_base = _quadrant_config.get('coord_base', 100)
        x_weights = _quadrant_config.get('x_weights', [0.5, 0.5])
        y_weights = _quadrant_config.get('y_weights', [0.3, 0.7])
        rounding_precision = _quadrant_config.get('rounding_precision', 2)

        # Use deterministic hash (MD5) instead of Python's hash()
        # Python's hash() is non-deterministic (randomized seed per process)
        content_hash = int(hashlib.md5(content.encode('utf-8')).hexdigest()[:16], 16)

        # Calculate x coordinate (v+)
        x_base = ((content_hash % coord_base) / float(coord_base))
        v_plus = (x_base * x_weights[0]) + (confidence * x_weights[1])

        # Calculate y coordinate (v-)
        y_base = (((content_hash >> 8) % coord_base) / float(coord_base))
        v_minus = (y_base * y_weights[0]) + (confidence * y_weights[1])

        return QuadrantCoordinates(
            v_plus=round(v_plus, rounding_precision),
            v_minus=round(v_minus, rounding_precision)
        )

    def compute_binary_vote(self, confidence: float, vote_threshold: float = None) -> int:
        """
        RCP v4 Equation 1: Convert continuous confidence to binary vote.

        v(ω, a) ∈ {0, 1}
        v(ω, a) = 1 if confidence >= threshold, else 0

        Args:
            confidence: Continuous confidence score [0, 1]
            vote_threshold: Threshold for binary decision (default from config)

        Returns:
            1 if confidence >= threshold, else 0
        """
        if vote_threshold is None:
            vote_threshold = _consensus_config.get('vote_threshold', 0.5)
        return 1 if confidence >= vote_threshold else 0
