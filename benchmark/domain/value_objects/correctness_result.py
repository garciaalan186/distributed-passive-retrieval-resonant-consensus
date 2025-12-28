"""
Value object representing correctness evaluation results.

This is a pure data structure with no I/O dependencies.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CorrectnessResult:
    """
    Result of evaluating response correctness.

    Attributes:
        is_correct: Whether the response contains the expected answer
        correctness_score: Quantitative score from 0.0 (wrong) to 1.0 (perfect)
        matched_entities: Entities from the response that matched expectations
        expected_entities: Entities that were expected to be present
        explanation: Human-readable explanation of the evaluation
    """
    is_correct: bool
    correctness_score: float  # 0.0 to 1.0
    matched_entities: List[str]
    expected_entities: List[str]
    explanation: str

    def __post_init__(self):
        """Validate invariants."""
        if not 0.0 <= self.correctness_score <= 1.0:
            raise ValueError(f"correctness_score must be in [0.0, 1.0], got {self.correctness_score}")
