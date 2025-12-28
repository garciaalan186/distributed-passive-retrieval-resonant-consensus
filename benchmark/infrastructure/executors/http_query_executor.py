"""
HTTP Query Executor

This is a legacy executor that wraps the existing HTTP-based query logic
from research_benchmark.py. It's useful for:

1. Cloud deployments where benchmark runs separately from DPR-RC services
2. Testing deployed systems end-to-end
3. Backward compatibility during migration

This executor should be used when:
- Benchmarking a deployed Cloud Run service
- Testing HTTP interface performance
- Running benchmarks from a different machine than DPR-RC
"""

import time
import asyncio
import aiohttp
from typing import Optional, List

from benchmark.domain.interfaces import IQueryExecutor, QueryExecutionResult


class HTTPQueryExecutor(IQueryExecutor):
    """
    Legacy HTTP-based query executor.

    This wraps the HTTP call logic from the original research_benchmark.py
    but implements the IQueryExecutor interface for clean architecture.

    Key Differences from DPRRCQueryExecutor:
    - DPRRCQueryExecutor: Will use in-process use cases (future)
    - HTTPQueryExecutor: Always uses HTTP (cloud deployments)

    Both measure latency the same way for comparable results.
    """

    def __init__(
        self,
        controller_url: str,
        timeout: float = 60.0
    ):
        """
        Initialize HTTP query executor.

        Args:
            controller_url: Full URL to the query endpoint
            timeout: Request timeout in seconds
        """
        # Ensure URL ends with /query
        if not controller_url.endswith("/query"):
            self._controller_url = f"{controller_url.rstrip('/')}/query"
        else:
            self._controller_url = controller_url

        self._timeout = timeout

    async def execute(
        self,
        query: str,
        query_id: str,
        timestamp_context: Optional[str] = None
    ) -> QueryExecutionResult:
        """
        Execute a single query via HTTP.

        This implements the exact same HTTP call as research_benchmark.py
        but returns a structured QueryExecutionResult.

        Args:
            query: The query text
            query_id: Unique identifier for tracing
            timestamp_context: Optional temporal context

        Returns:
            QueryExecutionResult with response and metrics
        """
        start_time = time.time()

        try:
            # Prepare request payload (matches research_benchmark.py format)
            payload = {
                "query_text": query,
                "trace_id": query_id,
            }

            if timestamp_context:
                payload["timestamp_context"] = timestamp_context

            # Execute HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._controller_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    # Measure latency
                    latency_ms = (time.time() - start_time) * 1000

                    if response.status == 200:
                        data = await response.json()

                        # Extract fields (same as research_benchmark.py)
                        return QueryExecutionResult(
                            query_id=query_id,
                            query_text=query,
                            response=data.get("final_answer", ""),
                            confidence=data.get("confidence", 0.0),
                            latency_ms=latency_ms,
                            success=True,
                            metadata={
                                "status": data.get("status"),
                                "sources": data.get("sources", []),
                                "superposition": data.get("superposition"),
                                "trace_id": query_id
                            }
                        )
                    else:
                        # HTTP error
                        error_text = await response.text()
                        return QueryExecutionResult(
                            query_id=query_id,
                            query_text=query,
                            response="",
                            confidence=0.0,
                            latency_ms=latency_ms,
                            success=False,
                            error=f"HTTP {response.status}: {error_text[:200]}"
                        )

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return QueryExecutionResult(
                query_id=query_id,
                query_text=query,
                response="",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"Timeout after {self._timeout}s"
            )

        except aiohttp.ClientError as e:
            # Network/connection errors
            latency_ms = (time.time() - start_time) * 1000
            return QueryExecutionResult(
                query_id=query_id,
                query_text=query,
                response="",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"HTTP Client Error: {str(e)}"
            )

        except Exception as e:
            # All other errors
            latency_ms = (time.time() - start_time) * 1000
            return QueryExecutionResult(
                query_id=query_id,
                query_text=query,
                response="",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"{type(e).__name__}: {str(e)}"
            )

    async def execute_batch(
        self,
        queries: List[tuple[str, str]],
        timestamp_context: Optional[str] = None
    ) -> List[QueryExecutionResult]:
        """
        Execute multiple queries concurrently via HTTP.

        Args:
            queries: List of (query_id, query_text) tuples
            timestamp_context: Optional temporal context for all queries

        Returns:
            List of QueryExecutionResult in same order as input
        """
        tasks = [
            self.execute(query_text, query_id, timestamp_context)
            for query_id, query_text in queries
        ]
        return await asyncio.gather(*tasks)

    @property
    def executor_id(self) -> str:
        """Return identifier for this executor"""
        return "dprrc-http"

    async def close(self):
        """Close any resources (no-op for HTTP executor)"""
        pass

    async def __aenter__(self):
        """Support async context manager"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager"""
        await self.close()
