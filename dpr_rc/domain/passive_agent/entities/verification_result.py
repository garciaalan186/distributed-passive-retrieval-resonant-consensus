"""
Domain Entity: VerificationResult

Represents the result of L2 semantic verification performed by the SLM.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    """
    Result from L2 verification step.

    Per DPR-RC specification, verification is performed by SLM-based
    semantic analysis, NOT simple token overlap.
    """

    content: str
    confidence_score: float  # Raw confidence from SLM
    verified: bool  # Whether content passes verification threshold
    explanation: Optional[str] = None  # SLM explanation (optional)
    depth_penalty: float = 0.0  # Penalty factor for hierarchical depth (1/(1+i))

    @property
    def adjusted_confidence(self) -> float:
        """
        Confidence adjusted by depth penalty.

        Per RCP v4 Eq. 9: C(r_p) = V(q, context_p) × 1/(1+i)
        """
        return self.confidence_score * (1 / (1 + self.depth_penalty))
