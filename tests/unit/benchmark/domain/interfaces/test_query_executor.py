"""
Unit tests for QueryExecutionResult dataclass

Tests the core data structure used across all executors.
"""

import pytest
from benchmark.domain.interfaces import QueryExecutionResult


class TestQueryExecutionResult:
    """Test QueryExecutionResult dataclass validation and behavior"""

    def test_valid_result_creation(self):
        """Test creating a valid QueryExecutionResult"""
        result = QueryExecutionResult(
            query_id="test-123",
            query_text="What is DPR-RC?",
            response="DPR-RC is a distributed processing system",
            confidence=0.85,
            latency_ms=150.5,
            success=True
        )

        assert result.query_id == "test-123"
        assert result.query_text == "What is DPR-RC?"
        assert result.response == "DPR-RC is a distributed processing system"
        assert result.confidence == 0.85
        assert result.latency_ms == 150.5
        assert result.success is True
        assert result.error is None
        assert result.metadata == {}

    def test_result_with_metadata(self):
        """Test QueryExecutionResult with metadata"""
        metadata = {
            "status": "success",
            "sources": ["doc1", "doc2"],
            "superposition": {"perspectives": 3}
        }

        result = QueryExecutionResult(
            query_id="test-456",
            query_text="Test query",
            response="Test response",
            confidence=0.9,
            latency_ms=100.0,
            success=True,
            metadata=metadata
        )

        assert result.metadata == metadata
        assert result.metadata["sources"] == ["doc1", "doc2"]

    def test_failed_result_with_error(self):
        """Test QueryExecutionResult for failed execution"""
        result = QueryExecutionResult(
            query_id="test-789",
            query_text="Failing query",
            response="",
            confidence=0.0,
            latency_ms=50.0,
            success=False,
            error="Timeout after 60s"
        )

        assert result.success is False
        assert result.error == "Timeout after 60s"
        assert result.response == ""
        assert result.confidence == 0.0

    def test_confidence_validation_too_high(self):
        """Test that confidence > 1.0 raises ValueError"""
        with pytest.raises(ValueError, match="Confidence must be in"):
            QueryExecutionResult(
                query_id="test",
                query_text="test",
                response="test",
                confidence=1.5,  # Invalid: > 1.0
                latency_ms=100.0,
                success=True
            )

    def test_confidence_validation_negative(self):
        """Test that negative confidence raises ValueError"""
        with pytest.raises(ValueError, match="Confidence must be in"):
            QueryExecutionResult(
                query_id="test",
                query_text="test",
                response="test",
                confidence=-0.1,  # Invalid: < 0.0
                latency_ms=100.0,
                success=True
            )

    def test_latency_validation_negative(self):
        """Test that negative latency raises ValueError"""
        with pytest.raises(ValueError, match="Latency cannot be negative"):
            QueryExecutionResult(
                query_id="test",
                query_text="test",
                response="test",
                confidence=0.5,
                latency_ms=-10.0,  # Invalid: negative
                success=True
            )

    def test_edge_case_zero_confidence(self):
        """Test that 0.0 confidence is valid"""
        result = QueryExecutionResult(
            query_id="test",
            query_text="test",
            response="test",
            confidence=0.0,  # Valid edge case
            latency_ms=100.0,
            success=True
        )
        assert result.confidence == 0.0

    def test_edge_case_max_confidence(self):
        """Test that 1.0 confidence is valid"""
        result = QueryExecutionResult(
            query_id="test",
            query_text="test",
            response="test",
            confidence=1.0,  # Valid edge case
            latency_ms=100.0,
            success=True
        )
        assert result.confidence == 1.0

    def test_edge_case_zero_latency(self):
        """Test that 0.0 latency is valid (edge case)"""
        result = QueryExecutionResult(
            query_id="test",
            query_text="test",
            response="test",
            confidence=0.5,
            latency_ms=0.0,  # Valid edge case (theoretical)
            success=True
        )
        assert result.latency_ms == 0.0
