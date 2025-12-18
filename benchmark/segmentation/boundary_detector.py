"""
Boundary Detector.

Orchestrates multi-signal boundary detection by combining:
- Tempo-normalized IEI analysis
- Lexical coherence analysis
- Domain transition detection
- Rhythm variance analysis

Implements the integrated boundary decision logic from
tempo-normalized segmentation paper (Definition 7).
"""

from typing import List, Dict
import numpy as np

from .tempo_normalizer import TempoNormalizer, BoundaryCandidate
from .coherence_analyzer import CoherenceAnalyzer


class BoundaryDetector:
    """
    Detects natural discourse boundaries using multi-signal analysis.
    """

    def __init__(self,
                 theta_sim: float = 0.4,
                 theta_rhythm: float = 2.0,
                 tau_min: int = 86400,
                 k_iqr: float = 1.5):
        """
        Initialize boundary detector.

        Args:
            theta_sim: Coherence threshold
            theta_rhythm: Rhythm variance threshold
            tau_min: Minimum temporal gap (seconds)
            k_iqr: IQR multiplier for adaptive threshold
        """
        self.tempo_normalizer = TempoNormalizer(
            theta_sim=theta_sim,
            theta_rhythm=theta_rhythm,
            tau_min=tau_min,
            k_iqr=k_iqr
        )
        self.coherence_analyzer = CoherenceAnalyzer(
            coherence_threshold=theta_sim
        )

    def detect_boundaries(self,
                         events: List[Dict],
                         embeddings: Dict[str, np.ndarray]) -> List[BoundaryCandidate]:
        """
        Detect all boundary candidates using multi-signal analysis.

        Args:
            events: List of events sorted by timestamp
            embeddings: Event ID -> embedding vector mapping

        Returns:
            List of boundary candidates with signal details
        """
        # Use tempo normalizer to find all candidates
        boundaries = self.tempo_normalizer.find_candidate_boundaries(
            events,
            embeddings
        )

        # Sort by strength (strongest boundaries first)
        boundaries.sort(key=lambda b: b.strength, reverse=True)

        return boundaries

    def filter_weak_boundaries(self,
                              boundaries: List[BoundaryCandidate],
                              min_strength: float = 0.5) -> List[BoundaryCandidate]:
        """
        Filter out weak boundaries below minimum strength.

        Args:
            boundaries: All boundary candidates
            min_strength: Minimum strength threshold

        Returns:
            Filtered boundaries
        """
        return [b for b in boundaries if b.strength >= min_strength]

    def resolve_conflicts(self,
                         boundaries: List[BoundaryCandidate],
                         min_distance: int = 3) -> List[BoundaryCandidate]:
        """
        Resolve conflicting boundaries that are too close together.

        If multiple boundaries are within min_distance events,
        keep only the strongest one.

        Args:
            boundaries: Boundary candidates
            min_distance: Minimum distance between boundaries (in events)

        Returns:
            Resolved boundaries
        """
        if not boundaries:
            return []

        # Sort by index
        sorted_boundaries = sorted(boundaries, key=lambda b: b.index)

        resolved = []
        i = 0

        while i < len(sorted_boundaries):
            current = sorted_boundaries[i]

            # Find all boundaries within min_distance
            cluster = [current]
            j = i + 1

            while j < len(sorted_boundaries):
                if sorted_boundaries[j].index - current.index <= min_distance:
                    cluster.append(sorted_boundaries[j])
                    j += 1
                else:
                    break

            # Keep strongest in cluster
            strongest = max(cluster, key=lambda b: b.strength)
            resolved.append(strongest)

            i = j

        return resolved

    def validate_boundary(self,
                         boundary: BoundaryCandidate,
                         events: List[Dict],
                         embeddings: Dict[str, np.ndarray]) -> bool:
        """
        Validate a boundary candidate with additional checks.

        Args:
            boundary: Boundary to validate
            events: All events
            embeddings: Event embeddings

        Returns:
            True if boundary is valid
        """
        idx = boundary.index

        if idx <= 0 or idx >= len(events):
            return False

        # Check minimum strength
        if boundary.strength < 0.3:
            return False

        # At least one strong signal must fire
        signals = boundary.signals
        strong_signals = [
            signals.get('domain', False),
            signals.get('tempo', False) and signals.get('lexical', False)
        ]

        return any(strong_signals)

    def get_boundary_summary(self, boundary: BoundaryCandidate) -> Dict:
        """
        Get human-readable summary of boundary signals.

        Args:
            boundary: Boundary candidate

        Returns:
            Summary dictionary
        """
        signals = boundary.signals

        return {
            'index': boundary.index,
            'timestamp': boundary.timestamp,
            'strength': round(boundary.strength, 3),
            'signals_fired': [
                name for name, fired in signals.items()
                if isinstance(fired, bool) and fired
            ],
            'tempo_ratio': round(signals.get('rho', 0), 2),
            'coherence': round(signals.get('coherence', 0), 3),
            'normalized_gap': round(signals.get('delta_t_norm', 0), 1)
        }
