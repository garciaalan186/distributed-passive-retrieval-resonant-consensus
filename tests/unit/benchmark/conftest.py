"""
Shared test fixtures and utilities for benchmark tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


def create_mock_aiohttp_response(status=200, json_data=None, text_data=""):
    """
    Helper to create a properly mocked aiohttp response with async context managers.

    Args:
        status: HTTP status code
        json_data: Dictionary to return from response.json()
        text_data: String to return from response.text()

    Returns:
        Tuple of (mock_session_context, mock_response) ready to use with patch
    """
    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status = status

    if json_data is not None:
        mock_response.json = AsyncMock(return_value=json_data)
    if text_data:
        mock_response.text = AsyncMock(return_value=text_data)

    # Mock async context manager for session.post()
    mock_post_context = AsyncMock()
    mock_post_context.__aenter__.return_value = mock_response
    mock_post_context.__aexit__.return_value = None

    # Mock session
    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_context

    # Mock ClientSession as async context manager
    mock_session_context = AsyncMock()
    mock_session_context.__aenter__.return_value = mock_session
    mock_session_context.__aexit__.return_value = None

    return mock_session_context, mock_response


def create_mock_aiohttp_error(exception):
    """
    Helper to create a mock that raises an exception.

    Args:
        exception: Exception instance to raise

    Returns:
        mock_session_context ready to use with patch
    """
    # Mock session.post() to raise exception
    mock_session = MagicMock()
    mock_session.post.side_effect = exception

    # Mock ClientSession as async context manager
    mock_session_context = AsyncMock()
    mock_session_context.__aenter__.return_value = mock_session
    mock_session_context.__aexit__.return_value = None

    return mock_session_context
