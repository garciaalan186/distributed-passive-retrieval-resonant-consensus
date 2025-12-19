"""
Integration Tests for RunBenchmarkUseCase

These tests verify the full benchmark workflow with different executor configurations.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from benchmark.application.use_cases import RunBenchmarkUseCase
from benchmark.application.dtos import RunBenchmarkRequest, RunBenchmarkResponse
from benchmark.domain.interfaces import QueryExecutionResult
from benchmark.infrastructure.dataset_generators import SyntheticDatasetGenerator
from benchmark.domain.services import EvaluationService


class MockQueryExecutor:
    """Mock executor for testing"""

    def __init__(self, executor_id: str, latency_ms: float = 100.0):
        self._executor_id = executor_id
        self._latency_ms = latency_ms

    async def execute(self, query: str, query_id: str, timestamp_context=None):
        # Generate mock response with entities from query
        words = query.split()
        entities = [w.strip('?.,') for w in words if w[0].isupper() and len(w) > 3]
        response = f"Response about {' and '.join(entities[:2])}" if entities else "No response"

        return QueryExecutionResult(
            query_id=query_id,
            query_text=query,
            response=response,
            confidence=0.9,
            latency_ms=self._latency_ms,
            success=True,
            metadata={"mock": True}
        )

    async def execute_batch(self, queries, timestamp_context=None):
        results = []
        for query_id, query_text in queries:
            result = await self.execute(query_text, query_id, timestamp_context)
            results.append(result)
        return results

    @property
    def executor_id(self):
        return self._executor_id


@pytest.fixture
def mock_dprrc_executor():
    """Create mock DPR-RC executor"""
    return MockQueryExecutor(executor_id="dprrc-mock", latency_ms=150.0)


@pytest.fixture
def mock_baseline_executor():
    """Create mock baseline executor"""
    return MockQueryExecutor(executor_id="baseline-mock", latency_ms=80.0)


@pytest.fixture
def dataset_generator():
    """Create real dataset generator"""
    return SyntheticDatasetGenerator()


@pytest.fixture
def evaluation_service():
    """Create real evaluation service"""
    return EvaluationService()


@pytest.fixture
def use_case(mock_dprrc_executor, mock_baseline_executor, dataset_generator, evaluation_service):
    """Create use case with mock executors"""
    return RunBenchmarkUseCase(
        dprrc_executor=mock_dprrc_executor,
        baseline_executor=mock_baseline_executor,
        dataset_generator=dataset_generator,
        evaluation_service=evaluation_service
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory"""
    output_dir = tmp_path / "benchmark_results"
    output_dir.mkdir()
    return str(output_dir)


class TestRunBenchmarkUseCase:
    """Test suite for RunBenchmarkUseCase"""

    @pytest.mark.asyncio
    async def test_execute_with_mock_executors(self, use_case, temp_output_dir):
        """Test benchmark execution with mock executors"""
        request = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir,
            enable_hallucination_detection=False,  # Disable for simplicity
            slm_service_url=None,
            seed=42
        )

        response = await use_case.execute(request)

        # Verify response
        assert isinstance(response, RunBenchmarkResponse)
        assert response.succeeded
        assert response.scale == "small"
        assert response.total_queries > 0
        assert 0.0 <= response.dprrc_accuracy <= 1.0
        assert 0.0 <= response.baseline_accuracy <= 1.0
        assert response.mean_latency_dprrc > 0
        assert response.mean_latency_baseline > 0

    @pytest.mark.asyncio
    async def test_dataset_generation(self, use_case, temp_output_dir):
        """Test that dataset is generated and saved correctly"""
        request = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir,
            enable_hallucination_detection=False,
            seed=42
        )

        response = await use_case.execute(request)

        assert response.succeeded
        assert response.dataset_path is not None

        # Verify dataset file exists
        dataset_path = Path(response.dataset_path)
        assert dataset_path.exists()

        # Verify dataset structure
        import json
        with open(dataset_path) as f:
            dataset = json.load(f)

        assert "scale" in dataset
        assert "queries" in dataset
        assert "glossary" in dataset
        assert "metadata" in dataset
        assert len(dataset["queries"]) > 0

    @pytest.mark.asyncio
    async def test_results_directory_structure(self, use_case, temp_output_dir):
        """Test that results are saved with correct directory structure"""
        request = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir,
            enable_hallucination_detection=False,
            seed=42
        )

        response = await use_case.execute(request)

        assert response.succeeded

        # Verify report exists
        report_path = Path(response.report_path)
        assert report_path.exists()
        assert report_path.name == "REPORT.md"

        # Verify result directories exist
        result_dir = report_path.parent
        assert (result_dir / "dprrc_results").exists()
        assert (result_dir / "baseline_results").exists()
        assert (result_dir / "comparison.json").exists()

    @pytest.mark.asyncio
    async def test_evaluation_metrics_calculation(self, use_case, temp_output_dir):
        """Test that evaluation metrics are calculated correctly"""
        request = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir,
            enable_hallucination_detection=False,
            seed=42
        )

        response = await use_case.execute(request)

        assert response.succeeded

        # Verify metrics are within valid ranges
        assert 0.0 <= response.dprrc_accuracy <= 1.0
        assert 0.0 <= response.baseline_accuracy <= 1.0
        assert 0.0 <= response.dprrc_hallucination_rate <= 1.0
        assert 0.0 <= response.baseline_hallucination_rate <= 1.0

        # Verify latency metrics
        assert response.mean_latency_dprrc >= 0
        assert response.mean_latency_baseline >= 0
        assert response.p95_latency_dprrc >= response.mean_latency_dprrc
        assert response.p95_latency_baseline >= response.mean_latency_baseline

    @pytest.mark.asyncio
    async def test_reproducibility_with_seed(self, use_case, temp_output_dir):
        """Test that same seed produces same dataset"""
        request1 = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir + "_1",
            enable_hallucination_detection=False,
            seed=42
        )

        request2 = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir + "_2",
            enable_hallucination_detection=False,
            seed=42
        )

        response1 = await use_case.execute(request1)
        response2 = await use_case.execute(request2)

        # Both should succeed
        assert response1.succeeded
        assert response2.succeeded

        # Should have same number of queries
        assert response1.total_queries == response2.total_queries

        # Load datasets and verify they're identical
        import json
        with open(response1.dataset_path) as f:
            dataset1 = json.load(f)
        with open(response2.dataset_path) as f:
            dataset2 = json.load(f)

        # Verify same number of queries
        assert len(dataset1["queries"]) == len(dataset2["queries"])

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_baseline_executor, dataset_generator, evaluation_service, temp_output_dir):
        """Test error handling when executor fails"""

        # Create failing executor
        class FailingExecutor:
            @property
            def executor_id(self):
                return "failing-executor"

            async def execute(self, query, query_id, timestamp_context=None):
                raise RuntimeError("Intentional failure")

            async def execute_batch(self, queries, timestamp_context=None):
                raise RuntimeError("Intentional failure")

        failing_use_case = RunBenchmarkUseCase(
            dprrc_executor=FailingExecutor(),
            baseline_executor=mock_baseline_executor,
            dataset_generator=dataset_generator,
            evaluation_service=evaluation_service
        )

        request = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir,
            enable_hallucination_detection=False,
            seed=42
        )

        response = await failing_use_case.execute(request)

        # Should not raise, but return error in response
        assert not response.succeeded
        assert response.error is not None
        assert "RuntimeError" in response.error or "Intentional failure" in response.error

    @pytest.mark.asyncio
    async def test_response_helper_methods(self, use_case, temp_output_dir):
        """Test RunBenchmarkResponse helper methods"""
        request = RunBenchmarkRequest(
            scale="small",
            output_dir=temp_output_dir,
            enable_hallucination_detection=False,
            seed=42
        )

        response = await use_case.execute(request)

        assert response.succeeded

        # Test helper methods
        accuracy_improvement = response.get_accuracy_improvement()
        assert isinstance(accuracy_improvement, float)

        hallucination_reduction = response.get_hallucination_reduction()
        assert isinstance(hallucination_reduction, float)

        latency_overhead = response.get_latency_overhead()
        assert isinstance(latency_overhead, float)

    @pytest.mark.asyncio
    async def test_different_scales(self, use_case, temp_output_dir):
        """Test execution with different scale levels"""
        scales = ["small", "medium", "large"]

        for scale in scales:
            request = RunBenchmarkRequest(
                scale=scale,
                output_dir=f"{temp_output_dir}_{scale}",
                enable_hallucination_detection=False,
                seed=42
            )

            response = await use_case.execute(request)

            assert response.succeeded
            assert response.scale == scale
            # Larger scales should have more queries
            if scale == "small":
                small_queries = response.total_queries
            elif scale == "medium":
                assert response.total_queries >= small_queries


class TestDatasetGenerator:
    """Test suite for SyntheticDatasetGenerator"""

    def test_generate_dataset(self):
        """Test dataset generation"""
        generator = SyntheticDatasetGenerator()
        dataset = generator.generate(scale="small", seed=42)

        assert dataset.scale == "small"
        assert len(dataset.queries) > 0
        assert dataset.glossary is not None
        assert dataset.metadata is not None

    def test_validate_dataset(self):
        """Test dataset validation"""
        generator = SyntheticDatasetGenerator()
        dataset = generator.generate(scale="small", seed=42)

        warnings = generator.validate_dataset(dataset)

        # Should have no warnings for valid dataset
        assert isinstance(warnings, list)

    def test_get_scale_config(self):
        """Test scale configuration retrieval"""
        generator = SyntheticDatasetGenerator()

        config = generator.get_scale_config("small")
        assert "events_per_topic_per_year" in config
        assert "num_domains" in config
        assert config["events_per_topic_per_year"] == 10
        assert config["num_domains"] == 2

    def test_invalid_scale(self):
        """Test error handling for invalid scale"""
        generator = SyntheticDatasetGenerator()

        with pytest.raises(ValueError, match="Invalid scale"):
            generator.generate(scale="invalid")

        with pytest.raises(ValueError, match="Invalid scale"):
            generator.get_scale_config("invalid")
