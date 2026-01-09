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
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, List, Union

from benchmark.domain.interfaces import IQueryExecutor, QueryExecutionResult
from dpr_rc.application.use_cases import ProcessQueryUseCase
from dpr_rc.application.dtos import ProcessQueryRequest
from dpr_rc.infrastructure.services import SimpleRouterService, DirectSLMService, DirectWorkerService


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

    # Persistent thread pool for multi-GPU execution (shared across batches)
    _gpu_thread_pool: Optional[ThreadPoolExecutor] = None
    _gpu_process_pool: Optional[mp.Pool] = None
    _pool_lock = threading.Lock()

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
        queries: List[tuple],
    ) -> List[QueryExecutionResult]:
        """
        Execute multiple queries concurrently.

        Works in both HTTP and UseCase modes.

        Note on Concurrency:
        - In multi-GPU mode: Uses ThreadPoolExecutor for true parallel execution
          with GPU context assignment (each thread gets a different GPU)
        - Otherwise: Uses asyncio.gather() for async concurrent execution

        Args:
            queries: List of tuples, either:
                     - (query_id, query_text) for backward compatibility
                     - (query_id, query_text, timestamp_context) for per-query context

        Returns:
            List of QueryExecutionResult in same order as input
        """
        # Check if multi-GPU parallel execution is enabled
        multi_gpu = os.getenv("ENABLE_MULTI_GPU_WORKERS", "false").lower() == "true"

        if multi_gpu and self._mode == "usecase":
            # Multi-GPU mode: Use ThreadPoolExecutor for true parallel GPU execution
            return await self._execute_batch_multi_gpu(queries)
        else:
            # Standard async execution (single GPU or HTTP mode)
            tasks = []
            for q in queries:
                if len(q) >= 3:
                    query_id, query_text, timestamp_context = q[0], q[1], q[2]
                else:
                    query_id, query_text = q[0], q[1]
                    timestamp_context = None
                tasks.append(self.execute(query_text, query_id, timestamp_context))
            return await asyncio.gather(*tasks)

    @classmethod
    def _get_gpu_process_pool(cls, num_workers: int, num_gpus: int) -> mp.Pool:
        """
        Get or create persistent GPU process pool.

        Each worker process is initialized with a specific GPU assignment
        via CUDA_VISIBLE_DEVICES environment variable.

        Args:
            num_workers: Number of worker processes to create
            num_gpus: Number of physical GPUs to distribute workers across
        """
        if cls._gpu_process_pool is None:
            with cls._pool_lock:
                if cls._gpu_process_pool is None:
                    print(f"Creating process pool with {num_workers} workers across {num_gpus} GPUs...")

                    # Use spawn to get clean processes (required for CUDA)
                    ctx = mp.get_context("spawn")

                    # Capture critical environment variables to pass to spawned workers
                    # These are needed for routing and data access
                    env_vars = {
                        "LOCAL_DATASET_PATH": os.getenv("LOCAL_DATASET_PATH", ""),
                        "CHROMA_DB_PATH": os.getenv("CHROMA_DB_PATH", ""),
                        "SLM_MODEL": os.getenv("SLM_MODEL", ""),
                        "SLM_FAST_MODEL": os.getenv("SLM_FAST_MODEL", ""),
                        "SLM_USE_4BIT_QUANTIZATION": os.getenv("SLM_USE_4BIT_QUANTIZATION", ""),
                        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", ""),
                        "USE_DIRECT_SERVICES": os.getenv("USE_DIRECT_SERVICES", ""),
                    }

                    # Create pool with initializer for each worker
                    cls._gpu_process_pool = ctx.Pool(
                        processes=num_workers,
                        initializer=cls._init_gpu_worker,
                        initargs=(num_gpus, env_vars)  # Pass GPU count and env vars
                    )
                    print("Process pool created and workers initialized")

        return cls._gpu_process_pool

    @staticmethod
    def _init_gpu_worker(num_gpus: int, env_vars: dict = None):
        """
        Initialize a worker process with GPU assignment and environment variables.

        Called once when each worker process starts. Uses the worker's
        process index modulo num_gpus for round-robin GPU assignment.

        Args:
            num_gpus: Number of GPUs to distribute workers across
            env_vars: Dictionary of environment variables to set in worker
        """
        import os

        # Set environment variables passed from parent process
        # This is critical for spawned processes which don't inherit env vars
        if env_vars:
            for key, value in env_vars.items():
                if value:  # Only set non-empty values
                    os.environ[key] = value

        # Get worker identity from process name or use round-robin
        worker_name = mp.current_process().name
        if "SpawnPoolWorker-" in worker_name:
            worker_idx = int(worker_name.split("-")[-1]) - 1
        else:
            worker_idx = 0

        # Round-robin assignment: worker 0,2,4... → GPU 0; worker 1,3,5... → GPU 1
        gpu_id = worker_idx % num_gpus

        # Set CUDA visibility for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["ENABLE_MULTI_GPU_WORKERS"] = "false"  # Single GPU per process

        print(f"Worker {worker_name} (idx={worker_idx}) assigned to GPU {gpu_id}")
        if env_vars and env_vars.get("LOCAL_DATASET_PATH"):
            print(f"Worker {worker_name}: LOCAL_DATASET_PATH={env_vars.get('LOCAL_DATASET_PATH')}")

        # Pre-load the model
        from dpr_rc.infrastructure.slm import SLMFactory
        print(f"Worker {worker_name}: Loading model on GPU {gpu_id}...")
        SLMFactory.get_engine()
        print(f"Worker {worker_name}: Model ready")

    async def _execute_batch_multi_gpu(
        self,
        queries: List[tuple],
    ) -> List[QueryExecutionResult]:
        """
        Execute queries in parallel using ProcessPoolExecutor.

        Each query is processed by a worker with its own GPU and model instance.
        This avoids asyncio overhead and enables true parallel GPU execution.

        Args:
            queries: List of tuples, either:
                     - (query_id, query_text) for backward compatibility
                     - (query_id, query_text, timestamp_context) for per-query context
        """
        num_workers = int(os.getenv("NUM_WORKER_THREADS", "2"))
        num_gpus = int(os.getenv("NUM_GPUS", "2"))  # Actual GPU count
        pool = self._get_gpu_process_pool(num_workers, num_gpus)

        # Prepare args for each query with per-query timestamp_context
        work_items = []
        for q in queries:
            if len(q) >= 3:
                query_id, query_text, timestamp_context = q[0], q[1], q[2]
            else:
                query_id, query_text = q[0], q[1]
                timestamp_context = None
            work_items.append((query_text, query_id, timestamp_context))

        # Execute in parallel using process pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: pool.map(_process_query_sync, work_items)
        )

        # Convert worker results to QueryExecutionResult
        return [
            QueryExecutionResult(
                query_id=r.query_id,
                query_text=r.query_text,
                response=r.response,
                confidence=r.confidence,
                latency_ms=r.latency_ms,
                success=r.success,
                error=r.error,
                metadata=r.metadata
            )
            for r in results
        ]

    @classmethod
    def shutdown_pools(cls):
        """Shutdown process and thread pools."""
        with cls._pool_lock:
            if cls._gpu_process_pool is not None:
                cls._gpu_process_pool.close()
                cls._gpu_process_pool.join()
                cls._gpu_process_pool = None
            if cls._gpu_thread_pool is not None:
                cls._gpu_thread_pool.shutdown(wait=True)
                cls._gpu_thread_pool = None

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
        # UseCase mode: create use case with direct services (local benchmarking)
        slm_service = DirectSLMService()
        worker_service = DirectWorkerService()
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


# Dataclass for serializable worker results (must be module-level for pickling)
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class _WorkerResult:
    """Serializable result from GPU worker process."""
    query_id: str
    query_text: str
    response: str
    confidence: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _process_query_sync(args: tuple) -> _WorkerResult:
    """
    Process a single query synchronously in a worker process.

    This is a module-level function for multiprocessing pickling.
    Each worker process has its own model loaded on its assigned GPU.

    Args:
        args: Tuple of (query_text, query_id, timestamp_context)

    Returns:
        _WorkerResult with query execution results
    """
    query_text, query_id, timestamp_context = args
    start_time = time.time()

    try:
        # Import here to ensure fresh imports in spawned process
        from dpr_rc.infrastructure.services.direct_services import DirectSLMService, DirectWorkerService
        from dpr_rc.infrastructure.services import SimpleRouterService
        from dpr_rc.application.use_cases import ProcessQueryUseCase
        from dpr_rc.application.dtos import ProcessQueryRequest
        import asyncio

        # Create services (model already loaded in worker init)
        slm_service = DirectSLMService()
        router_service = SimpleRouterService()
        worker_service = DirectWorkerService()

        # Create use case
        use_case = ProcessQueryUseCase(
            slm_service=slm_service,
            router_service=router_service,
            worker_service=worker_service
        )

        # Create request
        request = ProcessQueryRequest(
            query_text=query_text,
            trace_id=query_id,
            timestamp_context=timestamp_context if timestamp_context else None,
            enable_query_enhancement=True
        )

        # Execute (run async in this process's event loop)
        response = asyncio.run(use_case.execute(request))

        latency_ms = (time.time() - start_time) * 1000

        return _WorkerResult(
            query_id=query_id,
            query_text=query_text,
            response=response.final_answer,
            confidence=response.confidence,
            latency_ms=latency_ms,
            success=(response.status in ["SUCCESS", "NO_DATA"]),
            metadata={
                "execution_mode": "process",
                "status": response.status,
                "sources": response.sources,
                "superposition": response.superposition
            }
        )

    except Exception as e:
        import traceback
        latency_ms = (time.time() - start_time) * 1000
        return _WorkerResult(
            query_id=query_id,
            query_text=query_text,
            response="",
            confidence=0.0,
            latency_ms=latency_ms,
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            metadata={"execution_mode": "process"}
        )
