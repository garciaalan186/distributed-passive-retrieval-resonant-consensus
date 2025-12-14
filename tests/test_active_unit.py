import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient
from dpr_rc.active_agent import app, RouteLogic, RFI_STREAM, VOTE_STREAM
from dpr_rc.models import QueryRequest, ConsensusVote

client = TestClient(app)


def test_route_logic():
    """Test time-sharded routing logic"""
    req = QueryRequest(query_text="foo", timestamp_context="2022-01-01", trace_id="abc")
    shards = RouteLogic.get_target_shards(req)
    assert "shard_2022" in shards

    # Test broadcast when no timestamp
    req_no_ts = QueryRequest(query_text="bar", trace_id="def")
    shards_broadcast = RouteLogic.get_target_shards(req_no_ts)
    assert "broadcast" in shards_broadcast


def test_query_broadcast():
    """Test that a query broadcasts an RFI to Redis"""
    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_pubsub = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub
        mock_pubsub.get_message.return_value = None  # No votes

        response = client.post("/query", json={
            "query_text": "testing query",
            "timestamp_context": "2020-01-01",
            "trace_id": "test_trace_1"
        })

        # Verify RFI was added to stream
        mock_redis.xadd.assert_called_once()
        args = mock_redis.xadd.call_args[0]
        assert args[0] == RFI_STREAM
        payload = args[1]
        assert payload["query_text"] == "testing query"
        assert payload["trace_id"] == "test_trace_1"


def test_consensus_logic_superposition():
    """Test that consensus logic properly classifies votes into quadrants"""
    votes_data = [
        # 3 votes for "Fact A" with high confidence (should reach Consensus)
        ConsensusVote(
            trace_id="t1", worker_id="w1", content_hash="h1",
            confidence_score=0.9, semantic_quadrant=[0.9, 0.9],
            content_snippet="Fact A"
        ).model_dump(),
        ConsensusVote(
            trace_id="t1", worker_id="w2", content_hash="h1",
            confidence_score=0.92, semantic_quadrant=[0.9, 0.9],
            content_snippet="Fact A"
        ).model_dump(),
        ConsensusVote(
            trace_id="t1", worker_id="w3", content_hash="h1",
            confidence_score=0.88, semantic_quadrant=[0.9, 0.9],
            content_snippet="Fact A"
        ).model_dump(),

        # 1 vote for "Fact B" with lower confidence (Perspective)
        ConsensusVote(
            trace_id="t1", worker_id="w4", content_hash="h2",
            confidence_score=0.65, semantic_quadrant=[0.2, 0.5],
            content_snippet="Fact B"
        ).model_dump(),
    ]

    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_pubsub = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub

        # Create message responses with proper 'data' field
        messages = [{"data": json.dumps(v)} for v in votes_data]
        # Return messages then None repeatedly
        mock_pubsub.get_message.side_effect = messages + [None] * 100

        response = client.post("/query", json={
            "query_text": "What is truth?",
            "trace_id": "t1"
        })

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "SUCCESS"
        # Check that Fact A is in the answer (consensus)
        assert "Fact A" in result["final_answer"]
        # Verify high confidence due to consensus
        assert result["confidence"] >= 0.9


def test_no_votes_returns_failed():
    """Test that no votes results in FAILED status"""
    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_pubsub = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub
        mock_pubsub.get_message.return_value = None  # No votes

        response = client.post("/query", json={
            "query_text": "Any query",
            "trace_id": "test_no_votes"
        })

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "FAILED"
        assert result["confidence"] == 0.0


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_perspectival_only_response():
    """Test response when we have perspectives but no strong consensus"""
    votes_data = [
        # 2 votes for different content with medium confidence
        ConsensusVote(
            trace_id="t2", worker_id="w1", content_hash="h1",
            confidence_score=0.55, semantic_quadrant=[0.5, 0.5],
            content_snippet="Perspective X"
        ).model_dump(),
        ConsensusVote(
            trace_id="t2", worker_id="w2", content_hash="h2",
            confidence_score=0.52, semantic_quadrant=[0.4, 0.6],
            content_snippet="Perspective Y"
        ).model_dump(),
    ]

    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_pubsub = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub

        messages = [{"data": json.dumps(v)} for v in votes_data]
        mock_pubsub.get_message.side_effect = messages + [None] * 100

        response = client.post("/query", json={
            "query_text": "What are the perspectives?",
            "trace_id": "t2"
        })

        assert response.status_code == 200
        result = response.json()

        # Should succeed with perspectives
        assert result["status"] == "SUCCESS"
        # Confidence should be moderate (not high consensus)
        assert result["confidence"] == 0.7
        # Should mention perspectives
        assert "perspectives" in result["final_answer"].lower() or "Perspective" in result["final_answer"]
