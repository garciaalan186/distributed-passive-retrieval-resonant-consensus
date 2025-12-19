"""
Simplified unit tests for Query Executors

These tests verify the executors work correctly with minimal mocking complexity.
"""

import pytest
from benchmark.domain.interfaces import QueryExecutionResult
from benchmark.infrastructure.executors import DPRRCQueryExecutor, HTTPQueryExecutor


class TestExecutorBasics:
    """Test basic executor functionality without complex mocking"""

    def test_dprrc_executor_initialization(self):
        """Test DPRRCQueryExecutor initializes correctly"""
        executor = DPRRCQueryExecutor(
            controller_url="http://localhost:8080",
            timeout=30.0
        )
        # Updated for Phase 2: executor_id now includes mode suffix
        assert executor.executor_id == "dprrc-http"  # HTTP mode by default
        assert executor._controller_url == "http://localhost:8080"
        assert executor._timeout == 30.0

    def test_dprrc_executor_url_normalization(self):
        """Test URL trailing slash is removed"""
        executor = DPRRCQueryExecutor("http://localhost:8080/")
        assert executor._controller_url == "http://localhost:8080"

    def test_http_executor_initialization(self):
        """Test HTTPQueryExecutor initializes correctly"""
        executor = HTTPQueryExecutor(
            controller_url="http://example.com/query",
            timeout=60.0
        )
        assert executor.executor_id == "dprrc-http"
        assert executor._controller_url == "http://example.com/query"
        assert executor._timeout == 60.0

    def test_http_executor_adds_query_suffix(self):
        """Test /query suffix is added if missing"""
        executor = HTTPQueryExecutor("http://example.com")
        assert executor._controller_url == "http://example.com/query"

    def test_http_executor_query_suffix_not_duplicated(self):
        """Test /query suffix is not duplicated"""
        executor = HTTPQueryExecutor("http://example.com/query")
        assert executor._controller_url == "http://example.com/query"


class TestQueryExecutionResult:
    """Test QueryExecutionResult dataclass"""

    def test_create_successful_result(self):
        """Test creating a successful result"""
        result = QueryExecutionResult(
            query_id="test-1",
            query_text="What is X?",
            response="X is Y",
            confidence=0.9,
            latency_ms=100.0,
            success=True
        )

        assert result.query_id == "test-1"
        assert result.success is True
        assert result.error is None

    def test_create_failed_result(self):
        """Test creating a failed result"""
        result = QueryExecutionResult(
            query_id="test-2",
            query_text="Query",
            response="",
            confidence=0.0,
            latency_ms=50.0,
            success=False,
            error="Timeout"
        )

        assert result.success is False
        assert result.error == "Timeout"

    def test_confidence_validation(self):
        """Test confidence must be in [0, 1]"""
        with pytest.raises(ValueError, match="Confidence must be in"):
            QueryExecutionResult(
                query_id="test",
                query_text="test",
                response="test",
                confidence=1.5,  # Invalid
                latency_ms=100.0,
                success=True
            )

    def test_latency_validation(self):
        """Test latency cannot be negative"""
        with pytest.raises(ValueError, match="Latency cannot be negative"):
            QueryExecutionResult(
                query_id="test",
                query_text="test",
                response="test",
                confidence=0.5,
                latency_ms=-10.0,  # Invalid
                success=True
            )
