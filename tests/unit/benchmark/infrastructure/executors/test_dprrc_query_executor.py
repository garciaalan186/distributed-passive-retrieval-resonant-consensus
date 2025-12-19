"""
Unit tests for DPRRCQueryExecutor

Tests the DPR-RC executor with mocked HTTP responses.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from aiohttp import ClientError, ClientTimeout

from benchmark.infrastructure.executors import DPRRCQueryExecutor
from benchmark.domain.interfaces import QueryExecutionResult


@pytest.fixture
def executor():
    """Create a DPRRCQueryExecutor instance for testing"""
    return DPRRCQueryExecutor(
        controller_url="http://localhost:8080",
        timeout=30.0
    )


class TestDPRRCQueryExecutor:
    """Test DPRRCQueryExecutor implementation"""

    def test_executor_initialization(self):
        """Test executor initializes with correct parameters"""
        executor = DPRRCQueryExecutor(
            controller_url="http://localhost:9000",
            timeout=60.0
        )

        assert executor._controller_url == "http://localhost:9000"
        assert executor._timeout == 60.0
        # Updated for Phase 2: executor_id now includes mode suffix
        assert executor.executor_id == "dprrc-http"  # HTTP mode by default

    def test_executor_url_normalization(self):
        """Test that trailing slash is removed from URL"""
        executor = DPRRCQueryExecutor(
            controller_url="http://localhost:8080/"
        )
        assert executor._controller_url == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_execute_successful_query(self, executor):
        """Test successful query execution"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "final_answer": "Test answer",
            "confidence": 0.95,
            "status": "success",
            "sources": ["doc1", "doc2"],
            "superposition": {"perspectives": 2}
        })

        # Properly mock async context manager for session.post()
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_post_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_context

        # Mock ClientSession as async context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None

        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            result = await executor.execute(
                query="What is DPR-RC?",
                query_id="test-123",
                timestamp_context="2023-01-01"
            )

        assert isinstance(result, QueryExecutionResult)
        assert result.query_id == "test-123"
        assert result.query_text == "What is DPR-RC?"
        assert result.response == "Test answer"
        assert result.confidence == 0.95
        assert result.success is True
        assert result.error is None
        assert result.latency_ms > 0
        assert result.metadata["sources"] == ["doc1", "doc2"]

    @pytest.mark.asyncio
    async def test_execute_http_error(self, executor):
        """Test handling of HTTP error response"""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        # Properly mock async context manager for session.post()
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_post_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_context

        # Mock ClientSession as async context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None

        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            result = await executor.execute(
                query="Test query",
                query_id="test-error",
            )

        assert result.success is False
        assert "HTTP 500" in result.error
        assert result.response == ""
        assert result.confidence == 0.0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_timeout(self, executor):
        """Test handling of timeout"""
        # Mock async context manager that raises TimeoutError on __aenter__
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.side_effect = asyncio.TimeoutError()
        mock_post_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_context

        # Mock ClientSession as async context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None

        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            result = await executor.execute(
                query="Test query",
                query_id="test-timeout",
            )

        assert result.success is False
        assert "Timeout" in result.error
        assert result.response == ""
        assert result.confidence == 0.0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_network_error(self, executor):
        """Test handling of network errors"""
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(
            side_effect=ClientError("Connection failed")
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await executor.execute(
                query="Test query",
                query_id="test-network-error",
            )

        assert result.success is False
        assert result.error is not None
        assert result.response == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_execute_batch(self, executor):
        """Test batch execution of multiple queries"""
        # Mock successful responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "final_answer": "Answer",
            "confidence": 0.9,
            "status": "success",
            "sources": [],
        })

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        queries = [
            ("q1", "Query 1"),
            ("q2", "Query 2"),
            ("q3", "Query 3"),
        ]

        with patch('aiohttp.ClientSession', return_value=mock_session):
            results = await executor.execute_batch(queries)

        assert len(results) == 3
        assert all(isinstance(r, QueryExecutionResult) for r in results)
        assert results[0].query_id == "q1"
        assert results[1].query_id == "q2"
        assert results[2].query_id == "q3"

    @pytest.mark.asyncio
    async def test_execute_batch_preserves_order(self, executor):
        """Test that batch execution preserves query order"""
        # Create mock responses with different delays to test ordering
        async def mock_post(*args, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200
            # Extract query_id from payload
            query_id = kwargs['json']['trace_id']
            mock_response.json = AsyncMock(return_value={
                "final_answer": f"Answer for {query_id}",
                "confidence": 0.9,
                "status": "success",
                "sources": [],
            })
            return mock_response

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(side_effect=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        queries = [("q1", "Q1"), ("q2", "Q2"), ("q3", "Q3")]

        with patch('aiohttp.ClientSession', return_value=mock_session):
            results = await executor.execute_batch(queries)

        # Verify order is preserved
        assert results[0].query_id == "q1"
        assert results[1].query_id == "q2"
        assert results[2].query_id == "q3"

    @pytest.mark.asyncio
    async def test_execute_with_timestamp_context(self, executor):
        """Test that timestamp_context is passed correctly"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "final_answer": "Answer",
            "confidence": 0.9,
            "status": "success",
            "sources": [],
        })

        mock_post = AsyncMock(return_value=mock_response)
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            await executor.execute(
                query="Test",
                query_id="test",
                timestamp_context="2023-06-15"
            )

        # Verify timestamp_context was included in payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['timestamp_context'] == "2023-06-15"

    @pytest.mark.asyncio
    async def test_async_context_manager(self, executor):
        """Test using executor as async context manager"""
        async with executor as exec_instance:
            assert exec_instance is executor

    @pytest.mark.asyncio
    async def test_close(self, executor):
        """Test close method (currently a no-op)"""
        await executor.close()
        # Should not raise any errors
