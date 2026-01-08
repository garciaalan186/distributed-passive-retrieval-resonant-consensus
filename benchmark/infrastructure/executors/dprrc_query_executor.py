"""
DPR-RC Query Executor

This executor runs queries against the DPR-RC system with two modes:

1. **HTTP Mode** (default): Calls the active_agent HTTP endpoint
   - Used for cloud deployments
   - Compatible with deployed services
   - Includes HTTP overhead in measurements

2. **UseCase Mode** (USE_NEW_EXECUTOR=true): Calls ProcessQueryUseCase directly
   - Used for local benchmarks
   - Tests EXACT same code as production HTTP endpoint
   - Eliminates HTTP overhead from measurements
   - Ensures benchmark purity

Both modes test the SAME underlying business logic, just via different transport layers.
"""

import time
import asyncio
import aiohttp
import os
from typing import Optional, List, Union

from benchmark.domain.interfaces import IQueryExecutor, QueryExecutionResult
from dpr_rc.application.use_cases import ProcessQueryUseCase
from dpr_rc.application.dtos import ProcessQueryRequest
from dpr_rc.infrastructure.services import HttpSLMService, SimpleRouterService, HttpWorkerService


class DPRRCQueryExecutor(IQueryExecutor):
    """
    Executes queries against DPR-RC system.

    Supports two execution modes via dependency injection:
    - HTTP Mode: Uses aiohttp to call HTTP endpoint (for cloud benchmarks)
    - UseCase Mode: Uses ProcessQueryUseCase directly (for local benchmarks)

    Design Principles:
    1. **Same Logic**: Both modes execute identical business logic
    2. **Timing Accuracy**: Measures only query processing time (start to response)
    3. **Error Isolation**: All errors converted to QueryExecutionResult (never throws)
    4. **No Hidden State**: Each query is independent
    5. **Async Support**: Full async support for concurrent execution
    """

    def __init__(
        self,
        use_case: Optional[ProcessQueryUseCase] = None,
        controller_url: Optional[str] = None,
        timeout: float = 60.0,
        enable_query_enhancement: bool = True
    ):
        """
        Initialize the DPR-RC query executor.

        Args:
            use_case: Optional ProcessQueryUseCase instance (UseCase mode)
                     If provided, uses direct use case execution
                     If None, uses HTTP mode with controller_url
            controller_url: Base URL for HTTP mode (default: "http://localhost:8080")
                           Ignored if use_case is provided
            timeout: Request timeout in seconds
            enable_query_enhancement: Whether to enable SLM query enhancement

        Examples:
            # HTTP Mode (for cloud benchmarks)
            executor = DPRRCQueryExecutor(
                controller_url="https://my-service.run.app"
            )

            # UseCase Mode (for local benchmarks)
            use_case = ProcessQueryUseCase(
                slm_service=HttpSLMService(),
                router_service=SimpleRouterService(),
                worker_service=HttpWorkerService()
            )
            executor = DPRRCQueryExecutor(use_case=use_case)
        """
        self._use_case = use_case
        self._controller_url = (controller_url or "http://localhost:8080").rstrip('/')
        self._timeout = timeout
        self._enable_enhancement = enable_query_enhancement

        # Determine execution mode
        self._mode = "usecase" if use_case is not None else "http"

    async def execute(
        self,
        query: str,
        query_id: str,
        timestamp_context: Optional[str] = None
    ) -> QueryExecutionResult:
        """
        Execute a single query via the DPR-RC system.

        Routing:
        - If use_case is set: calls use case directly (UseCase mode)
        - Otherwise: makes HTTP request (HTTP mode)

        Timing Measurement:
        - Starts immediately before execution
        - Ends immediately after completion
        - Includes: query enhancement, routing, consensus, superposition
        - Excludes: HTTP overhead in UseCase mode (benchmark purity)

        Args:
            query: The query text
            query_id: Unique identifier for tracing
            timestamp_context: Optional temporal context (e.g., "2023-01-01")

        Returns:
            QueryExecutionResult with response and metrics
        """
        if self._mode == "usecase":
            return await self._execute_via_usecase(query, query_id, timestamp_context)
        else:
            return await self._execute_via_http(query, query_id, timestamp_context)

    async def _execute_via_usecase(
        self,
        query: str,
        query_id: str,
        timestamp_context: Optional[str] = None
    ) -> QueryExecutionResult:
        """
        Execute query via ProcessQueryUseCase (direct call).

        This mode eliminates HTTP overhead and tests the exact same
        code path as the production HTTP endpoint.
        """
        start_time = time.time()

        try:
            # Create request DTO
            request = ProcessQueryRequest(
                query_text=query,
                trace_id=query_id,
                timestamp_context=timestamp_context,
                enable_query_enhancement=self._enable_enhancement
            )

            # Execute use case directly
            response = await self._use_case.execute(request)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Convert to QueryExecutionResult
            return QueryExecutionResult(
                query_id=query_id,
                query_text=query,
                response=response.final_answer,
                confidence=response.confidence,
                latency_ms=latency_ms,
                success=(response.status in ["SUCCESS", "NO_DATA"]),
                error=response.metadata.get("error") if response.status == "ERROR" else None,
                metadata={
                    "status": response.status,
                    "sources": response.sources,
                    "superposition": response.superposition,
                    "trace_id": query_id,
                    "execution_mode": "usecase",
                    **response.metadata
                }
            )

        except Exception as e:
            # Shouldn't happen (use case doesn't raise), but handle defensively
            latency_ms = (time.time() - start_time) * 1000
            return QueryExecutionResult(
                query_id=query_id,
                query_text=query,
                response="",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={"execution_mode": "usecase"}
            )

    async def _execute_via_http(
        self,
        query: str,
        query_id: str,
        timestamp_context: Optional[str] = None
    ) -> QueryExecutionResult:
        """
        Execute query via HTTP endpoint (legacy mode).

        This mode is used for cloud deployments and includes HTTP overhead.
        """
        start_time = time.time()

        try:
            # Prepare request payload
            payload = {
                "query_text": query,
                "trace_id": query_id,
            }

            # Add timestamp context if provided
            if timestamp_context:
                payload["timestamp_context"] = timestamp_context

            # Execute query via HTTP
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._controller_url}/query",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    # Calculate latency immediately after response
                    latency_ms = (time.time() - start_time) * 1000

                    if response.status == 200:
                        data = await response.json()

                        return QueryExecutionResult(
                            query_id=query_id,
                            query_text=query,
                            response=data["final_answer"],
                            confidence=data["confidence"],
                            latency_ms=latency_ms,
                            success=True,
                            metadata={
                                "status": data["status"],
                                "sources": data.get("sources", []),
                                "resonance_matrix": data.get("resonance_matrix"),
                                "trace_id": query_id,
                                "execution_mode": "http"
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
                            error=f"HTTP {response.status}: {error_text}",
                            metadata={"execution_mode": "http"}
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
                error=f"Timeout after {self._timeout}s",
                metadata={"execution_mode": "http"}
            )

        except Exception as e:
            # All other errors (network, parsing, etc.)
            latency_ms = (time.time() - start_time) * 1000
            return QueryExecutionResult(
                query_id=query_id,
                query_text=query,
                response="",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={"execution_mode": "http"}
            )

    async def execute_batch(
        self,
        queries: List[tuple[str, str]],
        timestamp_context: Optional[str] = None
    ) -> List[QueryExecutionResult]:
        """
        Execute multiple queries concurrently.

        Works in both HTTP and UseCase modes.

        Note on Concurrency:
        - Uses asyncio.gather() for true concurrent execution
        - Individual query latencies are still measured accurately
        - Batch execution does not affect per-query timing

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
        return f"dprrc-{self._mode}"

    @property
    def execution_mode(self) -> str:
        """Return current execution mode (http or usecase)"""
        return self._mode

    async def close(self):
        """
        Close any resources.

        Note: Currently no persistent connections in either mode.
        Future implementations may need to close database connections, etc.
        """
        pass

    async def __aenter__(self):
        """Support async context manager"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager"""
        await self.close()


# Factory function for easy construction
def create_dprrc_executor(
    use_new_executor: bool = False,
    controller_url: Optional[str] = None,
    worker_url: Optional[str] = None,
    slm_url: Optional[str] = None,
    timeout: float = 60.0,
    enable_query_enhancement: bool = True
) -> DPRRCQueryExecutor:
    """
    Factory function to create DPRRCQueryExecutor with appropriate mode.

    Args:
        use_new_executor: If True, use UseCase mode; if False, use HTTP mode
        controller_url: URL for HTTP mode (ignored in UseCase mode)
        worker_url: Worker URL for UseCase mode (defaults to env var)
        slm_url: SLM URL for UseCase mode (defaults to env var)
        timeout: Request timeout in seconds
        enable_query_enhancement: Whether to enable SLM enhancement

    Returns:
        Configured DPRRCQueryExecutor instance

    Examples:
        # HTTP mode (default)
        executor = create_dprrc_executor(
            use_new_executor=False,
            controller_url="http://localhost:8080"
        )

        # UseCase mode
        executor = create_dprrc_executor(
            use_new_executor=True,
            worker_url="http://localhost:8082",
            slm_url="http://localhost:8081"
        )
    """
    if use_new_executor:
        # UseCase mode: create use case with service dependencies
        # Check if we should use direct (in-process) services
        use_direct_services = os.getenv("USE_DIRECT_SERVICES", "false").lower() == "true"

        if use_direct_services:
            # Direct mode: in-process services (for local benchmarking)
            from dpr_rc.infrastructure.services import DirectSLMService, DirectWorkerService
            slm_service = DirectSLMService()
            worker_service = DirectWorkerService()
        else:
            # HTTP mode: remote services (for distributed deployment)
            slm_service = HttpSLMService(
                slm_service_url=slm_url,
                timeout=timeout,
                enable_enhancement=enable_query_enhancement
            )
            worker_service = HttpWorkerService(
                worker_urls=worker_url,
                timeout=timeout
            )

        router_service = SimpleRouterService()

        use_case = ProcessQueryUseCase(
            slm_service=slm_service,
            router_service=router_service,
            worker_service=worker_service
        )

        return DPRRCQueryExecutor(
            use_case=use_case,
            timeout=timeout,
            enable_query_enhancement=enable_query_enhancement
        )
    else:
        # HTTP mode: use controller URL
        return DPRRCQueryExecutor(
            controller_url=controller_url,
            timeout=timeout,
            enable_query_enhancement=enable_query_enhancement
        )
