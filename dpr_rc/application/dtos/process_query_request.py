"""
ProcessQueryRequest DTO

Data Transfer Object for query processing requests.
Maps from external API contracts to internal use case inputs.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
import uuid


class ProcessQueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "What were the key ML advances in 2023?",
                "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp_context": "2023-01-01",
                "enable_query_enhancement": True
            }
        }
    )
    """
    Request DTO for ProcessQueryUseCase.

    This DTO represents the input contract for query processing,
    decoupling the use case from HTTP/API concerns.

    Attributes:
        query_text: The user's query text
        trace_id: Unique identifier for tracing this query through the system
        timestamp_context: Optional temporal context (e.g., "2023-01-01")
        enable_query_enhancement: Whether to use SLM for query enhancement
    """

    query_text: str = Field(..., description="The query text to process")
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trace ID for logging and debugging"
    )
    timestamp_context: Optional[str] = Field(
        None,
        description="Optional temporal context for time-sharded routing (e.g., '2023-01-01')"
    )
    enable_query_enhancement: bool = Field(
        True,
        description="Whether to enhance query via SLM service"
    )
