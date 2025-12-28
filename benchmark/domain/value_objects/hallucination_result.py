"""
Value object representing hallucination detection results.

This is a pure data structure with no I/O dependencies.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HallucinationResult:
    """
    Result of hallucination detection evaluation.

    Attributes:
        has_hallucination: Whether any hallucination was detected
        hallucination_type: Type of hallucination (e.g., "invalid_term", "factual_error")
        severity: Severity level - "none", "minor", "major", "critical"
        explanation: Human-readable explanation of what was detected
        flagged_content: Specific content flagged as potentially hallucinated
    """
    has_hallucination: bool
    hallucination_type: Optional[str]
    severity: str  # "none", "minor", "major", "critical"
    explanation: str
    flagged_content: List[str]

    VALID_SEVERITIES = {"none", "minor", "major", "critical"}

    def __post_init__(self):
        """Validate invariants."""
        if self.severity not in self.VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {self.VALID_SEVERITIES}, got '{self.severity}'"
            )

        if self.has_hallucination and self.hallucination_type is None:
            raise ValueError("hallucination_type must be set when has_hallucination is True")

        if not self.has_hallucination and self.severity != "none":
            raise ValueError("severity must be 'none' when has_hallucination is False")
