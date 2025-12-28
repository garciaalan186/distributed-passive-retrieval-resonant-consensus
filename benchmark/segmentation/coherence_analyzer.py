"""
Coherence Analyzer.

Computes lexical coherence signals for boundary detection using
embedding-based semantic similarity.

Adapts the lexical coherence signal from tempo-normalized segmentation
to work with pre-computed event embeddings.
"""

from typing import Dict, List
import numpy as np


class CoherenceAnalyzer:
    """
    Analyzes semantic coherence between events using embeddings.

    Lower coherence (similarity) indicates a potential topic shift
    and natural boundary location.
    """

    def __init__(self, coherence_threshold: float = 0.4):
        """
        Initialize coherence analyzer.

        Args:
            coherence_threshold: Threshold below which coherence is considered "broken"
        """
        self.coherence_threshold = coherence_threshold

    def compute_coherence(self,
                         event1_id: str,
                         event2_id: str,
                         embeddings: Dict[str, np.ndarray]) -> float:
        """
        Compute semantic coherence between two events.

        Uses cosine similarity of event embeddings as coherence measure.

        Args:
            event1_id: First event ID
            event2_id: Second event ID
            embeddings: Event ID -> embedding vector mapping

        Returns:
            Coherence score (0.0 - 1.0), higher = more coherent
        """
        if event1_id not in embeddings or event2_id not in embeddings:
            return 0.5  # Unknown, assume neutral coherence

        emb1 = embeddings[event1_id]
        emb2 = embeddings[event2_id]

        return self._cosine_similarity(emb1, emb2)

    def compute_coherence_sequence(self,
                                  event_ids: List[str],
                                  embeddings: Dict[str, np.ndarray]) -> List[float]:
        """
        Compute coherence scores for a sequence of events.

        Args:
            event_ids: List of event IDs in temporal order
            embeddings: Event embeddings

        Returns:
            List of coherence scores between consecutive events
        """
        if len(event_ids) < 2:
            return []

        coherence_scores = []
        for i in range(len(event_ids) - 1):
            score = self.compute_coherence(
                event_ids[i],
                event_ids[i + 1],
                embeddings
            )
            coherence_scores.append(score)

        return coherence_scores

    def is_coherence_break(self,
                          event1_id: str,
                          event2_id: str,
                          embeddings: Dict[str, np.ndarray]) -> bool:
        """
        Determine if coherence is broken between events.

        Args:
            event1_id: First event ID
            event2_id: Second event ID
            embeddings: Event embeddings

        Returns:
            True if coherence is below threshold (potential boundary)
        """
        coherence = self.compute_coherence(event1_id, event2_id, embeddings)
        return coherence < self.coherence_threshold

    def find_coherence_breaks(self,
                            event_ids: List[str],
                            embeddings: Dict[str, np.ndarray]) -> List[int]:
        """
        Find indices where coherence breaks occur.

        Args:
            event_ids: List of event IDs
            embeddings: Event embeddings

        Returns:
            List of indices where coherence drops below threshold
        """
        coherence_scores = self.compute_coherence_sequence(event_ids, embeddings)

        break_indices = []
        for i, score in enumerate(coherence_scores):
            if score < self.coherence_threshold:
                break_indices.append(i + 1)  # Boundary is at next event

        return break_indices

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (-1.0 to 1.0, normalized to 0.0-1.0)
        """
        if vec1 is None or vec2 is None:
            return 0.0

        # Ensure vectors are 1D
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0

    def compute_local_coherence_variance(self,
                                       event_ids: List[str],
                                       embeddings: Dict[str, np.ndarray],
                                       window_size: int = 5) -> List[float]:
        """
        Compute local coherence variance using sliding window.

        High variance indicates topic instability (potential boundaries).

        Args:
            event_ids: Event IDs in sequence
            embeddings: Event embeddings
            window_size: Size of sliding window

        Returns:
            List of variance scores (aligned to window centers)
        """
        coherence_scores = self.compute_coherence_sequence(event_ids, embeddings)

        if len(coherence_scores) < window_size:
            return []

        variances = []
        half_window = window_size // 2

        for i in range(half_window, len(coherence_scores) - half_window):
            window_start = i - half_window
            window_end = i + half_window + 1
            window_scores = coherence_scores[window_start:window_end]

            variance = np.var(window_scores) if window_scores else 0.0
            variances.append(variance)

        return variances
