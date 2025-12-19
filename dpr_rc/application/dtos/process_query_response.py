"""
ProcessQueryResponse DTO

Data Transfer Object for query processing responses.
Encapsulates all output from the ProcessQueryUseCase.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class ProcessQueryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                "final_answer": "The key ML advances in 2023 included...",
                "confidence": 0.95,
                "status": "SUCCESS",
                "sources": ["worker-2023-1", "worker-2023-2"],
                "superposition": {
                    "consensus_facts": ["Fact 1", "Fact 2"],
                    "perspectival_claims": []
                },
                "metadata": {
                    "enhanced_query": "machine learning advances 2023",
                    "num_votes": 2,
                    "num_consensus": 2,
                    "num_perspectival": 0
                }
            }
        }
    )
    """
    Response DTO for ProcessQueryUseCase.

    This DTO represents the output contract for query processing,
    containing all information needed to construct an API response
    or benchmark result.

    Attributes:
        trace_id: Same trace ID from request for correlation
        final_answer: The synthesized answer from consensus
        confidence: Confidence score (0.0 to 1.0)
        status: Processing status (SUCCESS, FAILED, NO_DATA)
        sources: List of worker IDs that contributed votes
        superposition: Optional superposition object with consensus and perspectives
        metadata: Additional metadata about processing
    """

    trace_id: str = Field(..., description="Trace ID from request")
    final_answer: str = Field(..., description="Synthesized answer from consensus")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    status: str = Field(
        ...,
        description="Processing status: SUCCESS, FAILED, NO_DATA"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Worker IDs that contributed to this answer"
    )
    superposition: Optional[Dict[str, Any]] = Field(
        None,
        description="Superposition object containing consensus facts and perspectival claims"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing metadata (enhancement info, timing, etc.)"
    )
