"""
DTO: Process RFI Response

Data Transfer Object for RFI processing results.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ProcessRFIResponse:
    """Response from processing RFI."""

    trace_id: str
    votes: List[Dict[str, Any]]
    worker_id: str
    cluster_id: str
    status: str  # "success", "no_results", "error"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for FastAPI response."""
        return {
            "trace_id": self.trace_id,
            "votes": self.votes,
            "worker_id": self.worker_id,
            "cluster_id": self.cluster_id,
            "status": self.status,
            "message": self.message,
        }
