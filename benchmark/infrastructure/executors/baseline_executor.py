"""
Baseline Query Executor

Executes queries against a baseline RAG system for comparison.

This executor wraps the PassiveWorker (naive RAG) for use in benchmarks.
It provides a clean IQueryExecutor interface for the baseline system.
"""

import time
import asyncio
from typing import Optional, List

from benchmark.domain.interfaces import IQueryExecutor, QueryExecutionResult


class BaselineExecutor(IQueryExecutor):
    """
    Executor for baseline RAG queries.

    This executor uses the PassiveWorker (naive RAG) as a baseline for
    comparison with DPR-RC. It provides the same interface as other executors
    but executes queries using simple retrieval without superposition.

    Design Principles:
    1. **Simple RAG**: Uses naive retrieval without consensus or superposition
    2. **Fair Comparison**: Measures only retrieval time (no HTTP overhead when local)
    3. **Same Interface**: Implements IQueryExecutor for consistency
    4. **Async Support**: Async interface even though PassiveWorker is sync

    Note:
        This executor requires the PassiveWorker to be available locally.
        For cloud benchmarks, use HTTPQueryExecutor pointing to a baseline endpoint.
    """

    def __init__(self, worker_url: Optional[str] = None, timeout: float = 60.0):
        """
        Initialize baseline executor.

        Args:
            worker_url: Optional worker URL (currently unused, for future HTTP support)
            timeout: Request timeout in seconds
        """
        self._worker_url = worker_url
        self._timeout = timeout
        self._worker = None

    def _ensure_worker(self):
        """Lazy-load the PassiveWorker"""
        if self._worker is None:
            try:
                from dpr_rc.passive_agent import PassiveWorker
                self._worker = PassiveWorker()
            except ImportError as e:
                raise RuntimeError(
                    "PassiveWorker not available. Install dpr-rc dependencies or use HTTP mode."
                ) from e

    async def execute(
        self,
        query: str,
        query_id: str,
        timestamp_context: Optional[str] = None
    ) -> QueryExecutionResult:
        """
        Execute a query via baseline RAG.

        Args:
            query: The query text
            query_id: Unique identifier for tracing
            timestamp_context: Optional temporal context (e.g., "2023-01-01")

        Returns:
            QueryExecutionResult with response and metrics
        """
        self._ensure_worker()

        start_time = time.time()

        try:
            # Convert timestamp_context to shard_id format
            # timestamp_context is like "2015-12-31", shard_id should be "shard_2015"
            if timestamp_context:
                year = timestamp_context.split("-")[0]
                shard_id = f"shard_{year}"
            else:
                shard_id = "shard_2020"  # default

            # Execute retrieval (sync call wrapped in async)
            doc = await asyncio.to_thread(
                self._worker.retrieve,
                query,
                shard_id,
                timestamp_context
            )

            latency_ms = (time.time() - start_time) * 1000

            if doc:
                return QueryExecutionResult(
                    query_id=query_id,
                    query_text=query,
                    response=doc.get("content", ""),
                    confidence=1.0,  # Baseline is always confident
                    latency_ms=latency_ms,
                    success=True,
                    metadata={
                        "shard_id": shard_id,
                        "execution_mode": "local"
                    }
                )
            else:
                return QueryExecutionResult(
                    query_id=query_id,
                    query_text=query,
                    response="",
                    confidence=0.0,
                    latency_ms=latency_ms,
                    success=False,
                    error="No document retrieved",
                    metadata={"execution_mode": "local"}
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return QueryExecutionResult(
                query_id=query_id,
                query_text=query,
                response="",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={"execution_mode": "local"}
            )

    async def execute_batch(
        self,
        queries: List[tuple[str, str]],
        timestamp_context: Optional[str] = None
    ) -> List[QueryExecutionResult]:
        """
        Execute multiple queries.

        Note: PassiveWorker is synchronous, so we execute sequentially.
        For true concurrency, use HTTP mode with multiple worker instances.

        Args:
            queries: List of (query_id, query_text) tuples
            timestamp_context: Optional temporal context for all queries

        Returns:
            List of QueryExecutionResult in same order as input
        """
        results = []
        for query_id, query_text in queries:
            result = await self.execute(query_text, query_id, timestamp_context)
            results.append(result)
        return results

    @property
    def executor_id(self) -> str:
        """Return identifier for this executor"""
        return "baseline-local"

    async def close(self):
        """Close any resources"""
        # PassiveWorker doesn't need cleanup
        pass

    async def __aenter__(self):
        """Support async context manager"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager"""
        await self.close()
