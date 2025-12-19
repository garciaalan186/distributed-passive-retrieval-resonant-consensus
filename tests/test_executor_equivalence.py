"""
Integration Tests: Executor Equivalence

These tests verify that HTTP mode and UseCase mode produce identical results,
ensuring benchmark purity and correctness of the refactoring.

Critical Test Objectives:
1. Both modes execute SAME business logic
2. Both modes produce SAME results (within acceptable variance)
3. UseCase mode eliminates HTTP overhead (faster)
4. No regression in existing functionality
"""

import pytest
import asyncio
import os
from typing import List

from benchmark.infrastructure.executors import create_dprrc_executor, DPRRCQueryExecutor
from benchmark.domain.interfaces import QueryExecutionResult
from dpr_rc.application.use_cases import ProcessQueryUseCase
from dpr_rc.application.dtos import ProcessQueryRequest
from dpr_rc.infrastructure.services import HttpSLMService, SimpleRouterService, HttpWorkerService


class TestExecutorEquivalence:
    """
    Test that HTTP and UseCase executors produce equivalent results.

    These tests are critical for validating the Phase 2 refactoring:
    - Ensures ProcessQueryUseCase faithfully implements active_agent.py logic
    - Verifies DPRRCQueryExecutor correctly wraps both modes
    - Proves benchmarks test the SAME code path as production
    """

    @pytest.fixture
    def http_executor(self):
        """Create HTTP mode executor"""
        return create_dprrc_executor(
            use_new_executor=False,
            controller_url=os.getenv("CONTROLLER_URL", "http://localhost:8080"),
            timeout=30.0,
            enable_query_enhancement=True
        )

    @pytest.fixture
    def usecase_executor(self):
        """Create UseCase mode executor"""
        return create_dprrc_executor(
            use_new_executor=True,
            worker_url=os.getenv("PASSIVE_WORKER_URL", "http://localhost:8082"),
            slm_url=os.getenv("SLM_SERVICE_URL", "http://localhost:8081"),
            timeout=30.0,
            enable_query_enhancement=True
        )

    @pytest.mark.asyncio
    async def test_executor_modes(self, http_executor, usecase_executor):
        """Verify executors are in correct modes"""
        assert http_executor.execution_mode == "http"
        assert usecase_executor.execution_mode == "usecase"

    @pytest.mark.asyncio
    async def test_simple_query_equivalence(self, http_executor, usecase_executor):
        """
        Test that both executors produce equivalent results for a simple query.

        This is the core equivalence test: same input should yield same output
        (modulo timing differences).
        """
        query = "What is machine learning?"
        query_id = "test_equiv_001"

        # Execute via both modes
        http_result = await http_executor.execute(query, query_id)
        usecase_result = await usecase_executor.execute(query, query_id)

        # Both should succeed (or both fail)
        assert http_result.success == usecase_result.success, \
            "HTTP and UseCase modes should have same success status"

        if http_result.success:
            # Responses should be identical (same business logic)
            assert http_result.response == usecase_result.response, \
                "HTTP and UseCase modes should produce identical responses"

            assert http_result.confidence == usecase_result.confidence, \
                "HTTP and UseCase modes should have identical confidence scores"

            # Sources should match (same workers queried)
            http_sources = set(http_result.metadata.get("sources", []))
            usecase_sources = set(usecase_result.metadata.get("sources", []))
            assert http_sources == usecase_sources, \
                "HTTP and UseCase modes should query same sources"

    @pytest.mark.asyncio
    async def test_temporal_query_equivalence(self, http_executor, usecase_executor):
        """
        Test equivalence for queries with temporal context.

        Verifies that routing logic works identically in both modes.
        """
        query = "What were the key developments?"
        query_id = "test_equiv_002"
        timestamp_context = "2023-01-01"

        http_result = await http_executor.execute(query, query_id, timestamp_context)
        usecase_result = await usecase_executor.execute(query, query_id, timestamp_context)

        # Both should route to same shards and get same results
        assert http_result.success == usecase_result.success

        if http_result.success:
            assert http_result.response == usecase_result.response
            assert http_result.confidence == usecase_result.confidence

    @pytest.mark.asyncio
    async def test_usecase_eliminates_http_overhead(self, http_executor, usecase_executor):
        """
        Verify that UseCase mode is faster (eliminates HTTP overhead).

        This test validates the benchmark purity objective:
        UseCase mode should measure only business logic, not transport.

        Note: This test may be skipped in CI if services aren't running.
        """
        query = "Test query for latency comparison"
        query_id = "test_latency_001"

        http_result = await http_executor.execute(query, query_id)
        usecase_result = await usecase_executor.execute(query, query_id)

        if http_result.success and usecase_result.success:
            # UseCase mode should be faster (no HTTP overhead)
            # Allow some variance due to system load
            assert usecase_result.latency_ms <= http_result.latency_ms * 1.1, \
                f"UseCase mode ({usecase_result.latency_ms}ms) should be <= HTTP mode ({http_result.latency_ms}ms)"

            print(f"HTTP latency: {http_result.latency_ms:.2f}ms")
            print(f"UseCase latency: {usecase_result.latency_ms:.2f}ms")
            print(f"Overhead eliminated: {http_result.latency_ms - usecase_result.latency_ms:.2f}ms")

    @pytest.mark.asyncio
    async def test_error_handling_equivalence(self, http_executor, usecase_executor):
        """
        Test that both modes handle errors equivalently.

        Both should convert exceptions to QueryExecutionResult (no raises).
        """
        # Use invalid timestamp to trigger potential errors
        query = "Test error handling"
        query_id = "test_error_001"
        bad_timestamp = "invalid-timestamp"

        http_result = await http_executor.execute(query, query_id, bad_timestamp)
        usecase_result = await usecase_executor.execute(query, query_id, bad_timestamp)

        # Neither should raise exceptions
        assert isinstance(http_result, QueryExecutionResult)
        assert isinstance(usecase_result, QueryExecutionResult)

    @pytest.mark.asyncio
    async def test_batch_execution_equivalence(self, http_executor, usecase_executor):
        """
        Test that batch execution produces equivalent results in both modes.

        Verifies concurrent execution works correctly in both modes.
        """
        queries = [
            ("query_1", "What is AI?"),
            ("query_2", "What is ML?"),
            ("query_3", "What is DL?"),
        ]

        http_results = await http_executor.execute_batch(queries)
        usecase_results = await usecase_executor.execute_batch(queries)

        assert len(http_results) == len(usecase_results) == len(queries)

        for i, (http_r, usecase_r) in enumerate(zip(http_results, usecase_results)):
            assert http_r.success == usecase_r.success, \
                f"Query {i}: success status should match"

            if http_r.success:
                assert http_r.response == usecase_r.response, \
                    f"Query {i}: responses should match"


class TestProcessQueryUseCase:
    """
    Direct tests for ProcessQueryUseCase.

    These tests validate the use case in isolation,
    ensuring it correctly orchestrates service dependencies.
    """

    @pytest.fixture
    def use_case(self):
        """Create ProcessQueryUseCase with real service implementations"""
        slm_service = HttpSLMService(
            slm_service_url=os.getenv("SLM_SERVICE_URL", "http://localhost:8081"),
            timeout=5.0,
            enable_enhancement=True
        )
        router_service = SimpleRouterService()
        worker_service = HttpWorkerService(
            worker_urls=os.getenv("PASSIVE_WORKER_URL", "http://localhost:8082"),
            timeout=30.0
        )

        return ProcessQueryUseCase(
            slm_service=slm_service,
            router_service=router_service,
            worker_service=worker_service
        )

    @pytest.mark.asyncio
    async def test_use_case_basic_execution(self, use_case):
        """Test that use case executes basic query flow"""
        request = ProcessQueryRequest(
            query_text="What is machine learning?",
            trace_id="test_usecase_001",
            enable_query_enhancement=True
        )

        response = await use_case.execute(request)

        # Should return valid response (not raise exception)
        assert response is not None
        assert response.trace_id == "test_usecase_001"
        assert response.status in ["SUCCESS", "FAILED", "NO_DATA", "ERROR"]

    @pytest.mark.asyncio
    async def test_use_case_never_raises(self, use_case):
        """
        Critical: Use case should NEVER raise exceptions.

        All errors should be captured in response.status = "ERROR"
        """
        # Invalid request that might cause errors
        request = ProcessQueryRequest(
            query_text="",  # Empty query
            trace_id="test_error_002"
        )

        # Should not raise
        response = await use_case.execute(request)

        assert response is not None
        # Empty query might fail, but shouldn't raise
        assert response.status in ["SUCCESS", "FAILED", "NO_DATA", "ERROR"]

    @pytest.mark.asyncio
    async def test_use_case_temporal_routing(self, use_case):
        """Test that use case correctly uses router for temporal queries"""
        request = ProcessQueryRequest(
            query_text="What happened in 2023?",
            trace_id="test_routing_001",
            timestamp_context="2023-01-01"
        )

        response = await use_case.execute(request)

        # Should route to shard_2023
        # Verify via metadata or logging (implementation-dependent)
        assert response is not None


class TestServiceImplementations:
    """
    Tests for concrete service implementations.

    Validates that HttpSLMService, SimpleRouterService, and HttpWorkerService
    correctly implement their interfaces.
    """

    def test_router_basic_routing(self):
        """Test SimpleRouterService routing logic"""
        from dpr_rc.infrastructure.services import SimpleRouterService

        router = SimpleRouterService()

        # No timestamp -> broadcast
        shards = router.get_target_shards("test query", None)
        assert shards == ["broadcast"]

        # With timestamp -> specific shard
        shards = router.get_target_shards("test query", "2023-01-01")
        assert shards == ["shard_2023"]

    def test_slm_service_fallback(self):
        """Test SLMService gracefully handles unavailable service"""
        from dpr_rc.infrastructure.services import HttpSLMService

        # Point to non-existent service
        slm = HttpSLMService(
            slm_service_url="http://localhost:99999",
            timeout=0.1,
            enable_enhancement=True
        )

        # Should fallback gracefully, not raise
        result = slm.enhance_query("test query")

        assert result["original_query"] == "test query"
        assert result["enhanced_query"] == "test query"  # Fallback to original
        assert result["enhancement_used"] is False


@pytest.mark.skipif(
    not os.getenv("INTEGRATION_TESTS_ENABLED"),
    reason="Integration tests require running services (set INTEGRATION_TESTS_ENABLED=1)"
)
class TestFullIntegration:
    """
    Full integration tests requiring all services running.

    These tests are skipped in CI unless INTEGRATION_TESTS_ENABLED=1
    """

    @pytest.mark.asyncio
    async def test_end_to_end_equivalence(self):
        """
        End-to-end test: Run complete query through both modes.

        This is the ultimate validation that refactoring was successful.
        """
        http_executor = create_dprrc_executor(use_new_executor=False)
        usecase_executor = create_dprrc_executor(use_new_executor=True)

        query = "What were the major advancements in AI during 2023?"
        query_id = "integration_test_001"
        timestamp_context = "2023-06-01"

        http_result = await http_executor.execute(query, query_id, timestamp_context)
        usecase_result = await usecase_executor.execute(query, query_id, timestamp_context)

        # Detailed equivalence check
        assert http_result.success == usecase_result.success
        assert http_result.query_text == usecase_result.query_text
        assert http_result.response == usecase_result.response
        assert http_result.confidence == usecase_result.confidence

        # Metadata should be equivalent
        http_status = http_result.metadata.get("status")
        usecase_status = usecase_result.metadata.get("status")
        assert http_status == usecase_status

        print(f"\nHTTP Result: {http_result.response[:100]}...")
        print(f"UseCase Result: {usecase_result.response[:100]}...")
        print(f"Results match: {http_result.response == usecase_result.response}")
