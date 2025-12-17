"""
Tempo-Normalized Boundary Detection.

Adapts tempo-normalized segmentation methodology from the research paper
to historical event streams. Finds natural discourse boundaries based on:
- Normalized inter-event intervals (IEI)
- Tempo ratio (event clustering density)
- Rhythm variance (temporal pattern changes)

Based on: tempo_normalized_segmentation_enhanced.pdf
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque
import statistics
import numpy as np


@dataclass
class BoundaryCandidate:
    """A candidate boundary location with supporting signals."""
    index: int  # Event index where boundary occurs
    timestamp: str  # Timestamp of boundary
    strength: float  # Boundary strength (0.0 - 1.0)
    signals: Dict[str, bool]  # Which signals fired
    shifted_from: Optional[int] = None  # If boundary was shifted for causality


class TempoNormalizer:
    """
    Finds candidate boundaries using tempo-normalized segmentation.

    Adaptation for historical events:
    - Actor -> Research domain / perspective
    - Message length -> Event content token count
    - Response time -> Inter-event temporal gap
    - Lexical coherence -> Embedding cosine similarity
    - Session markers -> Domain transitions, claim type changes
    """

    def __init__(self,
                 theta_sim: float = 0.4,
                 theta_rhythm: float = 2.0,
                 tau_min: int = 86400,
                 k_iqr: float = 1.5,
                 history_window: int = 20):
        """
        Initialize tempo normalizer.

        Args:
            theta_sim: Coherence threshold (cosine similarity)
            theta_rhythm: Rhythm variance threshold (standard deviation)
            tau_min: Minimum temporal gap in seconds (default: 1 day)
            k_iqr: IQR multiplier for adaptive threshold
            history_window: Rolling window size for tempo history
        """
        self.theta_sim = theta_sim
        self.theta_rhythm = theta_rhythm
        self.tau_min = tau_min
        self.k_iqr = k_iqr
        self.history_window = history_window

        # Domain velocity (expected events per day)
        # Will be computed from data
        self.beta_domain: Dict[str, float] = {}

    def find_candidate_boundaries(self,
                                 events: List[Dict],
                                 embeddings: Dict[str, np.ndarray]) -> List[BoundaryCandidate]:
        """
        Find candidate boundaries using multi-signal tempo-normalized analysis.

        Args:
            events: List of events sorted by timestamp
            embeddings: Event ID -> embedding vector mapping

        Returns:
            List of boundary candidates with signals
        """
        if len(events) < 2:
            return []

        # Compute domain velocities (expected event rate)
        self._compute_domain_velocities(events)

        boundaries = []
        history = deque(maxlen=self.history_window)  # Tempo ratio history

        for i in range(1, len(events)):
            e_prev = events[i - 1]
            e_curr = events[i]

            # 1. Compute raw temporal interval
            delta_t_raw = self._compute_time_delta(
                e_prev.get('timestamp', ''),
                e_curr.get('timestamp', '')
            )

            # 2. Compute expected time based on domain velocity
            domain = e_curr.get('topic', 'unknown')
            expected_gap = self._compute_expected_gap(e_prev, domain)

            # 3. Normalize temporal interval
            delta_t_norm = max(0.0, delta_t_raw - expected_gap)

            # 4. Compute tempo ratio
            epsilon = 1.0  # Prevent division by zero
            rho = delta_t_raw / (expected_gap + epsilon)

            # 5. Lexical coherence signal (embedding similarity)
            coh = self._compute_coherence(
                e_prev.get('id', ''),
                e_curr.get('id', ''),
                embeddings
            )
            lexical_break = (coh < self.theta_sim)

            # 6. Domain transition signal
            domain_break = (
                e_prev.get('topic', '') != e_curr.get('topic', '')
                or e_prev.get('perspective', '') != e_curr.get('perspective', '')
            )

            # 7. Rhythm variance signal
            history.append(rho)
            rhythm_var = statistics.stdev(history) if len(history) > 1 else 0.0
            rhythm_break = (rhythm_var > self.theta_rhythm)

            # 8. Adaptive tempo threshold
            tau_adaptive = self._compute_adaptive_threshold(history)
            tau = max(self.tau_min, tau_adaptive)
            tempo_break = (delta_t_norm >= tau) or (rho >= 3.0)

            # 9. Multi-signal boundary decision (Definition 7 from paper)
            is_boundary = (
                domain_break  # Strong signal
                or (tempo_break and lexical_break)  # Validated tempo
                or (tempo_break and delta_t_norm > tau * 2)  # Absolute threshold
                or (lexical_break and rhythm_break)  # Topic + rhythm
            )

            if is_boundary:
                strength = self._compute_boundary_strength(
                    tempo_break, lexical_break, rhythm_break, domain_break
                )

                boundaries.append(BoundaryCandidate(
                    index=i,
                    timestamp=e_curr.get('timestamp', ''),
                    strength=strength,
                    signals={
                        'tempo': tempo_break,
                        'lexical': lexical_break,
                        'rhythm': rhythm_break,
                        'domain': domain_break,
                        'delta_t_norm': delta_t_norm,
                        'rho': rho,
                        'coherence': coh
                    }
                ))

        return boundaries

    def _compute_domain_velocities(self, events: List[Dict]) -> None:
        """
        Compute expected event rate per domain (beta_domain).

        Args:
            events: All events
        """
        domain_counts = {}
        domain_time_spans = {}

        for event in events:
            domain = event.get('topic', 'unknown')
            timestamp = event.get('timestamp', '')

            if domain not in domain_counts:
                domain_counts[domain] = 0
                domain_time_spans[domain] = {'min': timestamp, 'max': timestamp}

            domain_counts[domain] += 1
            if timestamp < domain_time_spans[domain]['min']:
                domain_time_spans[domain]['min'] = timestamp
            if timestamp > domain_time_spans[domain]['max']:
                domain_time_spans[domain]['max'] = timestamp

        # Compute velocity (events per second)
        for domain in domain_counts:
            time_span = self._compute_time_delta(
                domain_time_spans[domain]['min'],
                domain_time_spans[domain]['max']
            )
            if time_span > 0:
                self.beta_domain[domain] = time_span / domain_counts[domain]
            else:
                self.beta_domain[domain] = 86400.0  # Default: 1 day

    def _compute_expected_gap(self, event: Dict, domain: str) -> float:
        """
        Compute expected time gap based on domain velocity and event complexity.

        Args:
            event: Previous event
            domain: Current domain

        Returns:
            Expected gap in seconds
        """
        base_velocity = self.beta_domain.get(domain, 86400.0)

        # Complexity factor (longer events may have longer gaps after)
        content_length = len(event.get('content', ''))
        complexity_factor = 1.0 + (content_length / 10000.0)  # Normalize

        return base_velocity * complexity_factor

    def _compute_time_delta(self, ts1: str, ts2: str) -> float:
        """
        Compute time delta in seconds between timestamps.

        Args:
            ts1: Earlier timestamp (ISO format)
            ts2: Later timestamp (ISO format)

        Returns:
            Time delta in seconds
        """
        from datetime import datetime

        try:
            t1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
            return abs((t2 - t1).total_seconds())
        except Exception:
            return 0.0

    def _compute_coherence(self,
                          id1: str,
                          id2: str,
                          embeddings: Dict[str, np.ndarray]) -> float:
        """
        Compute lexical coherence via embedding cosine similarity.

        Args:
            id1: First event ID
            id2: Second event ID
            embeddings: Event embeddings

        Returns:
            Cosine similarity (0.0 - 1.0)
        """
        if id1 not in embeddings or id2 not in embeddings:
            return 0.5  # Unknown, assume neutral

        emb1 = embeddings[id1]
        emb2 = embeddings[id2]

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _compute_adaptive_threshold(self, history: deque) -> float:
        """
        Compute adaptive tempo threshold using IQR method.

        Args:
            history: Rolling window of tempo ratios

        Returns:
            Adaptive threshold
        """
        if len(history) < 4:
            return self.tau_min

        values = sorted(history)
        n = len(values)

        # Compute Q1, Q3, IQR
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = values[q1_idx]
        q3 = values[q3_idx]
        iqr = q3 - q1

        # Adaptive threshold: Q3 + k_iqr * IQR
        return q3 + self.k_iqr * iqr

    def _compute_boundary_strength(self,
                                   tempo_break: bool,
                                   lexical_break: bool,
                                   rhythm_break: bool,
                                   domain_break: bool) -> float:
        """
        Compute boundary strength based on signal consensus.

        Args:
            tempo_break: Tempo signal fired
            lexical_break: Lexical coherence break
            rhythm_break: Rhythm variance break
            domain_break: Domain transition

        Returns:
            Strength score (0.0 - 1.0)
        """
        signals = [tempo_break, lexical_break, rhythm_break, domain_break]
        weights = [0.3, 0.25, 0.2, 0.25]  # Domain gets higher weight

        strength = sum(w for s, w in zip(signals, weights) if s)
        return min(1.0, strength)
