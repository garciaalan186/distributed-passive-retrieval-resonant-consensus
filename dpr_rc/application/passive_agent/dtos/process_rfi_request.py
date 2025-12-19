"""
DTO: Process RFI Request

Data Transfer Object for incoming RFI requests.
"""

from typing import List, Optional
from dataclasses import dataclass
import json


@dataclass
class ProcessRFIRequest:
    """Request to process RFI from Active Agent."""

    trace_id: str
    query_text: str  # Enhanced query for retrieval
    original_query: str  # Original query for verification
    timestamp_context: str
    target_shards: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessRFIRequest":
        """Create from dictionary (FastAPI request body)."""
        trace_id = data.get("trace_id", "unknown")
        query_text = data.get("query_text", "")
        original_query = data.get("original_query", query_text)
        timestamp_context = data.get("timestamp_context", "")
        target_shards_str = data.get("target_shards", "[]")

        # Parse target shards - support both string and list formats
        try:
            if isinstance(target_shards_str, list):
                target_shards = target_shards_str
            elif target_shards_str:
                target_shards = json.loads(target_shards_str)
            else:
                target_shards = []
        except json.JSONDecodeError:
            target_shards = []

        return cls(
            trace_id=trace_id,
            query_text=query_text,
            original_query=original_query,
            timestamp_context=timestamp_context,
            target_shards=target_shards,
        )
