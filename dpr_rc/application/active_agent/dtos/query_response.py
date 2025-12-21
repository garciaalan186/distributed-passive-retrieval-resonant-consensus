"""
DTO: Query Response

Data Transfer Object for query processing results.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class QueryResponseDTO:
    """Response from processing a query."""

    trace_id: str
    final_answer: Optional[str]
    confidence: Optional[float]
    status: str
    sources: List[str]
    sources: List[str]
    resonance_matrix: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary for FastAPI response."""
        return {
            "trace_id": self.trace_id,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "status": self.status,
            "sources": self.sources,
            "resonance_matrix": self.resonance_matrix,
        }
