"""
Unit tests for HTTPQueryExecutor

Tests the legacy HTTP-based executor with mocked responses.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from benchmark.infrastructure.executors import HTTPQueryExecutor
from benchmark.domain.interfaces import QueryExecutionResult


@pytest.fixture
def executor():
    """Create HTTPQueryExecutor instance for testing"""
    return HTTPQueryExecutor(
        controller_url="http://example.com/query",
        timeout=30.0
    )


class TestHTTPQueryExecutor:
    """Test HTTPQueryExecutor implementation"""

    def test_executor_initialization(self):
        """Test executor initializes with correct parameters"""
        executor = HTTPQueryExecutor(
            controller_url="http://example.com/api/query",
            timeout=60.0
        )

        assert executor._controller_url == "http://example.com/api/query"
        assert executor._timeout == 60.0
        assert executor.executor_id == "dprrc-http"

    def test_executor_url_with_query_suffix(self):
        """Test that URL is correctly formed with /query suffix"""
        executor = HTTPQueryExecutor(
            controller_url="http://example.com"
        )
        assert executor._controller_url == "http://example.com/query"

    def test_executor_url_already_has_query(self):
        """Test that /query is not duplicated if already present"""
        executor = HTTPQueryExecutor(
            controller_url="http://example.com/query"
        )
        assert executor._controller_url == "http://example.com/query"

    @pytest.mark.asyncio
    async def test_execute_successful_query(self, executor):
        """Test successful query execution via HTTP"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "final_answer": "HTTP executor answer",
            "confidence": 0.88,
            "status": "completed",
            "sources": ["source1"],
            "superposition": None
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
                query="Test HTTP query",
                query_id="http-test-123",
                timestamp_context=None
            )

        assert isinstance(result, QueryExecutionResult)
        assert result.query_id == "http-test-123"
        assert result.query_text == "Test HTTP query"
        assert result.response == "HTTP executor answer"
        assert result.confidence == 0.88
        assert result.success is True
        assert result.error is None
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_handles_missing_fields(self, executor):
        """Test handling of response with missing optional fields"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            # Minimal response - some fields missing
            "final_answer": "Answer",
            "confidence": 0.5
            # Missing: status, sources, superposition
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
                query="Test",
                query_id="test",
            )

        assert result.success is True
        assert result.response == "Answer"
        assert result.confidence == 0.5
        assert result.metadata["sources"] == []
        assert result.metadata["status"] is None

    @pytest.mark.asyncio
    async def test_execute_http_404(self, executor):
        """Test handling of 404 Not Found"""
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Endpoint not found")

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
                query="Test",
                query_id="test-404",
            )

        assert result.success is False
        assert "HTTP 404" in result.error
        assert result.response == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_execute_timeout(self, executor):
        """Test handling of request timeout"""
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
                query="Test",
                query_id="test-timeout",
            )

        assert result.success is False
        assert "Timeout after 30.0s" in result.error
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_client_error(self, executor):
        """Test handling of aiohttp ClientError"""
        from aiohttp import ClientError

        # Mock async context manager that raises ClientError on __aenter__
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.side_effect = ClientError("DNS resolution failed")
        mock_post_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_context

        # Mock ClientSession as async context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None

        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            result = await executor.execute(
                query="Test",
                query_id="test-client-error",
            )

        assert result.success is False
        assert "HTTP Client Error" in result.error
        assert result.response == ""

    @pytest.mark.asyncio
    async def test_execute_generic_exception(self, executor):
        """Test handling of unexpected exceptions"""
        # Mock async context manager that raises ValueError on __aenter__
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.side_effect = ValueError("Unexpected error")
        mock_post_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_context

        # Mock ClientSession as async context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None

        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            result = await executor.execute(
                query="Test",
                query_id="test-exception",
            )

        assert result.success is False
        assert "ValueError" in result.error
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_execute_batch(self, executor):
        """Test batch execution returns results in order"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "final_answer": "Batch answer",
            "confidence": 0.8,
        })

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        queries = [
            ("batch-1", "Query 1"),
            ("batch-2", "Query 2"),
            ("batch-3", "Query 3"),
        ]

        with patch('aiohttp.ClientSession', return_value=mock_session):
            results = await executor.execute_batch(queries)

        assert len(results) == 3
        assert all(isinstance(r, QueryExecutionResult) for r in results)
        assert results[0].query_id == "batch-1"
        assert results[1].query_id == "batch-2"
        assert results[2].query_id == "batch-3"

    @pytest.mark.asyncio
    async def test_execute_batch_with_timestamp(self, executor):
        """Test batch execution with timestamp_context"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "final_answer": "Answer",
            "confidence": 0.9,
        })

        mock_post = AsyncMock(return_value=mock_response)
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        queries = [("q1", "Query 1")]

        with patch('aiohttp.ClientSession', return_value=mock_session):
            await executor.execute_batch(
                queries,
                timestamp_context="2024-01-01"
            )

        # Verify timestamp was passed
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['timestamp_context'] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_executor_id_is_correct(self, executor):
        """Test that executor_id is 'dprrc-http' (not 'dprrc')"""
        assert executor.executor_id == "dprrc-http"

    @pytest.mark.asyncio
    async def test_async_context_manager(self, executor):
        """Test async context manager support"""
        async with executor as exec_instance:
            assert exec_instance is executor

    @pytest.mark.asyncio
    async def test_close(self, executor):
        """Test close method"""
        await executor.close()
        # Should not raise errors

    @pytest.mark.asyncio
    async def test_error_message_truncation(self, executor):
        """Test that very long error messages are truncated"""
        long_error = "x" * 500  # Very long error message

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value=long_error)

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await executor.execute(
                query="Test",
                query_id="test",
            )

        # Error should be truncated to 200 chars
        assert result.success is False
        assert len(result.error) <= 250  # "HTTP 500: " + 200 chars + buffer
