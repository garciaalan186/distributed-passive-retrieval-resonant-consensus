"""
Query Executor Interface

This interface defines how benchmarks execute queries against different systems.
Implementations MUST NOT introduce confounding variables that would affect benchmark results.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class QueryExecutionResult:
    """
    Result from query execution.

    This dataclass captures all relevant metrics and metadata from a single query execution.
    All timing measurements should be in milliseconds for consistency.

    Attributes:
        query_id: Unique identifier for the query (for tracing)
        query_text: The original query text
        response: The system's response text
        confidence: Confidence score from 0.0 to 1.0
        latency_ms: End-to-end latency in milliseconds
        success: Whether the query executed successfully
        error: Error message if execution failed (None if successful)
        metadata: Additional system-specific metadata (e.g., superposition data, sources)
    """
    query_id: str
    query_text: str
    response: str
    confidence: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        """Validate invariants"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.latency_ms < 0:
            raise ValueError(f"Latency cannot be negative, got {self.latency_ms}")


class IQueryExecutor(ABC):
    """
    Interface for executing queries against a system.

    This interface is the foundation of benchmark decoupling. It allows benchmarks
    to test different systems (DPR-RC, baseline RAG, etc.) through a common interface
    without introducing implementation-specific bias.

    CRITICAL IMPLEMENTATION REQUIREMENTS:

    1. **Timing Purity**: Implementations MUST measure only the actual query processing time.
       Do NOT include:
       - Serialization/deserialization overhead
       - Network latency (unless that's what's being benchmarked)
       - Instrumentation code execution time

    2. **No Hidden State**: Each execute() call should be independent unless explicitly
       designed for stateful execution. Document any state management clearly.

    3. **Error Handling**: All exceptions MUST be caught and returned as QueryExecutionResult
       with success=False. Never let exceptions propagate to benchmarks.

    4. **Async Support**: Use async/await to enable concurrent execution in batch operations.
       This is essential for performance but must not affect measurement accuracy.

    Implementations:
    - DPRRCQueryExecutor: Uses ProcessQueryUseCase directly (no HTTP)
    - BaselineRAGQueryExecutor: Uses baseline RAG system
    - HTTPQueryExecutor: Legacy HTTP-based executor for cloud deployments
    - MockQueryExecutor: For testing
    """

    @abstractmethod
    async def execute(
        self,
        query: str,
        query_id: str,
        timestamp_context: Optional[str] = None
    ) -> QueryExecutionResult:
        """
        Execute a single query and return the result.

        Args:
            query: The query text to execute
            query_id: Unique identifier for this query (for tracing and correlation)
            timestamp_context: Optional temporal context for the query

        Returns:
            QueryExecutionResult with execution metrics and response

        Note:
            This method MUST be idempotent for the same query_id to enable
            reliable benchmarking and result verification.
        """
        pass

    @abstractmethod
    async def execute_batch(
        self,
        queries: List[tuple[str, str]],  # [(query_id, query_text), ...]
        timestamp_context: Optional[str] = None
    ) -> List[QueryExecutionResult]:
        """
        Execute multiple queries concurrently.

        Args:
            queries: List of (query_id, query_text) tuples
            timestamp_context: Optional temporal context for all queries

        Returns:
            List of QueryExecutionResult in the same order as input queries

        Note:
            Implementations SHOULD use asyncio.gather() or similar for true
            concurrency, but MUST ensure individual query latencies are measured
            accurately (not affected by concurrent execution overhead).
        """
        pass

    @property
    @abstractmethod
    def executor_id(self) -> str:
        """
        Return identifier for this executor.

        Returns:
            String identifier (e.g., 'dprrc', 'baseline', 'dprrc-http')

        Note:
            This identifier is used in benchmark reports and result storage.
            It MUST be unique and stable across benchmark runs.
        """
        pass
