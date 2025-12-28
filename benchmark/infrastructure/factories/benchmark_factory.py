"""
Benchmark Factory

Factory for creating fully-wired benchmark use cases with all dependencies.

This factory handles the complex dependency injection needed for benchmarks,
making it easy to create properly configured use cases in different contexts
(CLI, tests, notebooks, etc.).
"""

import os
from typing import Optional

from benchmark.application.use_cases import RunBenchmarkUseCase
from benchmark.domain.interfaces import IQueryExecutor, IDatasetGenerator
from benchmark.domain.services import EvaluationService
from benchmark.infrastructure.executors import create_dprrc_executor
from benchmark.infrastructure.dataset_generators import SyntheticDatasetGenerator


class BenchmarkFactory:
    """
    Factory to create fully-wired benchmark use cases.

    This factory provides a clean API for creating benchmarks with different
    configurations while hiding the complexity of dependency wiring.

    Design Principles:
    1. **Environment-Aware**: Uses environment variables for defaults
    2. **Explicit Overrides**: Allows explicit parameter overrides
    3. **Sensible Defaults**: Provides good defaults for common use cases
    4. **Multiple Executors**: Supports creating different executor types

    Environment Variables:
    - CONTROLLER_URL: DPR-RC controller URL (default: http://localhost:8080)
    - PASSIVE_WORKER_URL: Worker URL for UseCase mode (default: http://localhost:8082)
    - SLM_SERVICE_URL: SLM service URL (default: http://localhost:8081)
    - USE_NEW_EXECUTOR: Use UseCase mode instead of HTTP (default: false)
    """

    @staticmethod
    def create_benchmark_use_case(
        scale: str = "small",
        output_dir: str = "benchmark_results",
        use_new_executor: bool = False,
        controller_url: Optional[str] = None,
        worker_url: Optional[str] = None,
        slm_url: Optional[str] = None,
        baseline_mode: str = "http",
        timeout: float = 60.0,
        enable_query_enhancement: bool = True
    ) -> RunBenchmarkUseCase:
        """
        Create a fully-wired RunBenchmarkUseCase.

        Args:
            scale: Scale level ('small', 'medium', 'large', 'stress')
            output_dir: Directory to save benchmark results
            use_new_executor: If True, use ProcessQueryUseCase directly (UseCase mode)
                            If False, use HTTP mode (default)
            controller_url: DPR-RC controller URL (for HTTP mode)
            worker_url: Worker URL (for UseCase mode)
            slm_url: SLM service URL
            baseline_mode: Baseline executor mode ('http', 'local', 'mock')
            timeout: Request timeout in seconds
            enable_query_enhancement: Whether to enable SLM query enhancement

        Returns:
            Configured RunBenchmarkUseCase ready to execute

        Examples:
            # HTTP mode (default) - for cloud deployments
            use_case = BenchmarkFactory.create_benchmark_use_case(
                scale="small",
                controller_url="https://my-service.run.app"
            )

            # UseCase mode - for local benchmarks (benchmark purity)
            use_case = BenchmarkFactory.create_benchmark_use_case(
                scale="small",
                use_new_executor=True,
                worker_url="http://localhost:8082",
                slm_url="http://localhost:8081"
            )

            # Test mode with mocks
            use_case = BenchmarkFactory.create_benchmark_use_case(
                scale="small",
                baseline_mode="mock"
            )
        """
        # Use environment variables as defaults
        use_new_executor = use_new_executor or os.getenv("USE_NEW_EXECUTOR", "false").lower() == "true"
        controller_url = controller_url or os.getenv("CONTROLLER_URL", "http://localhost:8080")
        worker_url = worker_url or os.getenv("PASSIVE_WORKER_URL", "http://localhost:8082")
        slm_url = slm_url or os.getenv("SLM_SERVICE_URL", "http://localhost:8081")

        # Create DPR-RC executor
        dprrc_executor = create_dprrc_executor(
            use_new_executor=use_new_executor,
            controller_url=controller_url,
            worker_url=worker_url,
            slm_url=slm_url,
            timeout=timeout,
            enable_query_enhancement=enable_query_enhancement
        )

        # Create baseline executor
        baseline_executor = BenchmarkFactory._create_baseline_executor(
            mode=baseline_mode,
            controller_url=controller_url,
            worker_url=worker_url,
            timeout=timeout
        )

        # Create dataset generator
        dataset_generator = SyntheticDatasetGenerator()

        # Create evaluation service
        evaluation_service = EvaluationService()

        # Create and return use case
        return RunBenchmarkUseCase(
            dprrc_executor=dprrc_executor,
            baseline_executor=baseline_executor,
            dataset_generator=dataset_generator,
            evaluation_service=evaluation_service
        )

    @staticmethod
    def _create_baseline_executor(
        mode: str,
        controller_url: str,
        worker_url: str,
        timeout: float
    ) -> IQueryExecutor:
        """
        Create baseline executor based on mode.

        Args:
            mode: Executor mode ('http', 'local', 'mock')
            controller_url: Controller URL for HTTP mode
            worker_url: Worker URL for local mode
            timeout: Request timeout

        Returns:
            Configured baseline executor
        """
        if mode == "http":
            # Use HTTP executor pointing to baseline endpoint
            from benchmark.infrastructure.executors import HTTPQueryExecutor
            baseline_url = controller_url.replace("/query", "/baseline/query")
            return HTTPQueryExecutor(
                controller_url=baseline_url,
                timeout=timeout
            )

        elif mode == "local":
            # Use local baseline (PassiveWorker) via executor
            from benchmark.infrastructure.executors.baseline_executor import BaselineExecutor
            return BaselineExecutor(
                worker_url=worker_url,
                timeout=timeout
            )

        elif mode == "mock":
            # Use mock executor for testing
            from tests.unit.benchmark.conftest import MockQueryExecutor
            return MockQueryExecutor(executor_id="baseline-mock")

        else:
            raise ValueError(
                f"Invalid baseline_mode '{mode}'. "
                f"Valid options: 'http', 'local', 'mock'"
            )

    @staticmethod
    def create_dataset_generator(seed: Optional[int] = None) -> IDatasetGenerator:
        """
        Create a standalone dataset generator.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Configured dataset generator

        Example:
            generator = BenchmarkFactory.create_dataset_generator(seed=42)
            dataset = generator.generate(scale="small")
        """
        return SyntheticDatasetGenerator()

    @staticmethod
    def create_dprrc_executor(
        use_new_executor: bool = False,
        controller_url: Optional[str] = None,
        worker_url: Optional[str] = None,
        slm_url: Optional[str] = None,
        timeout: float = 60.0
    ) -> IQueryExecutor:
        """
        Create a standalone DPR-RC executor.

        Args:
            use_new_executor: Use UseCase mode instead of HTTP
            controller_url: Controller URL for HTTP mode
            worker_url: Worker URL for UseCase mode
            slm_url: SLM service URL
            timeout: Request timeout

        Returns:
            Configured DPR-RC executor

        Example:
            executor = BenchmarkFactory.create_dprrc_executor(
                use_new_executor=True,
                worker_url="http://localhost:8082"
            )
        """
        return create_dprrc_executor(
            use_new_executor=use_new_executor,
            controller_url=controller_url or os.getenv("CONTROLLER_URL", "http://localhost:8080"),
            worker_url=worker_url or os.getenv("PASSIVE_WORKER_URL", "http://localhost:8082"),
            slm_url=slm_url or os.getenv("SLM_SERVICE_URL", "http://localhost:8081"),
            timeout=timeout,
            enable_query_enhancement=True
        )
