"""
DTO: Query Request

Data Transfer Object for incoming query requests.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryRequestDTO:
    """Request to process a query."""

    trace_id: str
    query_text: str
    timestamp_context: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "QueryRequestDTO":
        """Create from dictionary (FastAPI request body)."""
        return cls(
            trace_id=data.get("trace_id", ""),
            query_text=data.get("query_text", ""),
            timestamp_context=data.get("timestamp_context"),
        )
