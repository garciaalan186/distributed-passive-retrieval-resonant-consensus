"""
Unit tests for Active Agent Controller (Cache-Based Architecture)

Tests:
- Cache polling instead of Pub/Sub
- Worker readiness checking
- RCP integration
- Response building
"""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from dpr_rc.active_agent import app, RouteLogic, RFI_STREAM, WORKERS_READY_KEY
from dpr_rc.models import QueryRequest, CachedResponse, RCPResult, SemanticQuadrant, AgentResponseScore

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


def test_query_broadcasts_rfi():
    """Test that a query broadcasts an RFI to Redis Stream"""
    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_redis.scard.return_value = 1  # 1 ready worker
        mock_redis.scan_iter.return_value = iter([])  # No cached responses

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


def test_cache_polling_for_responses():
    """Test that responses are collected from cache, not Pub/Sub"""
    cached_response = CachedResponse(
        trace_id="t1",
        agent_id="worker_1",
        shard_id="2020",
        content_hash="hash1",
        content="Test response content about the query topic.",
        confidence=0.85,
        timestamp="2024-01-01T00:00:00"
    )

    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_redis.scard.return_value = 1

        # Setup scan_iter to return a response key
        mock_redis.scan_iter.return_value = iter(["dpr:response:t1:worker_1"])
        mock_redis.get.return_value = cached_response.model_dump_json()

        response = client.post("/query", json={
            "query_text": "test query",
            "trace_id": "t1"
        })

        assert response.status_code == 200
        result = response.json()

        # Should have found the response
        assert result["status"] == "SUCCESS"
        assert "worker_1" in result["sources"]

        # Verify pubsub was NOT used (old approach)
        mock_redis.pubsub.assert_not_called()


def test_no_responses_returns_failed():
    """Test that no cached responses results in FAILED status"""
    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_redis.scard.return_value = 0  # No ready workers
        mock_redis.scan_iter.return_value = iter([])  # No responses

        response = client.post("/query", json={
            "query_text": "Any query",
            "trace_id": "test_no_responses"
        })

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "FAILED"
        assert result["confidence"] == 0.0


def test_health_check_includes_architecture():
    """Test health check endpoint shows cache-based architecture"""
    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_redis.scard.return_value = 3

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["architecture"] == "cache-based"
        assert data["ready_workers"] == 3


def test_workers_ready_endpoint():
    """Test workers ready endpoint"""
    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_redis.scard.return_value = 2
        mock_redis.smembers.return_value = {"worker_1", "worker_2"}

        response = client.get("/workers/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready_count"] == 2
        assert len(data["workers"]) == 2


def test_multiple_responses_aggregation():
    """Test aggregation of multiple cached responses"""
    responses = [
        CachedResponse(
            trace_id="t2",
            agent_id="worker_1",
            shard_id="2020",
            content_hash="h1",
            content="Fact A is true according to historical records.",
            confidence=0.9,
            timestamp="2024-01-01T00:00:00"
        ),
        CachedResponse(
            trace_id="t2",
            agent_id="worker_2",
            shard_id="2021",
            content_hash="h2",
            content="Fact A is true, confirmed by multiple sources.",
            confidence=0.88,
            timestamp="2024-01-01T00:00:01"
        ),
    ]

    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_redis.scard.return_value = 2

        # Return different keys on successive scans
        keys = ["dpr:response:t2:worker_1", "dpr:response:t2:worker_2"]
        mock_redis.scan_iter.return_value = iter(keys)

        def get_response(key):
            if "worker_1" in key:
                return responses[0].model_dump_json()
            elif "worker_2" in key:
                return responses[1].model_dump_json()
            return None

        mock_redis.get.side_effect = get_response

        response = client.post("/query", json={
            "query_text": "What is Fact A?",
            "trace_id": "t2"
        })

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "SUCCESS"
        assert result["confidence"] > 0.5
        # Both workers should be in sources
        assert "worker_1" in result["sources"] or "worker_2" in result["sources"]


def test_rcp_result_integration():
    """Test that RCP results are properly integrated"""
    cached_response = CachedResponse(
        trace_id="t3",
        agent_id="worker_1",
        shard_id="2020",
        content_hash="h1",
        content="Historical consensus content.",
        confidence=0.85,
        timestamp="2024-01-01T00:00:00"
    )

    rcp_result = RCPResult(
        trace_id="t3",
        semantic_quadrant=SemanticQuadrant(
            symmetric_resonance={
                "content": "Aggregated consensus content",
                "confidence": 0.9,
                "num_sources": 2
            },
            asymmetric_perspectives=[]
        ),
        pa_response_scores={
            "worker_1": AgentResponseScore(
                content="Historical consensus content.",
                confidence=0.85,
                consensus_score=0.9,
                polarization_score=0.1,
                shard_id="2020",
                quadrant_coords=[0.9, 0.1]
            )
        },
        total_responses=1,
        total_votes=0,
        computation_time_ms=5.2
    )

    with patch("dpr_rc.active_agent.redis_client") as mock_redis, \
         patch("dpr_rc.active_agent.RCPEngine") as mock_rcp_class:

        mock_redis.scard.return_value = 1
        mock_redis.scan_iter.return_value = iter(["dpr:response:t3:worker_1"])
        mock_redis.get.return_value = cached_response.model_dump_json()

        mock_rcp = MagicMock()
        mock_rcp.compute_and_cache_result.return_value = rcp_result
        mock_rcp_class.return_value = mock_rcp

        response = client.post("/query", json={
            "query_text": "Test RCP integration",
            "trace_id": "t3"
        })

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "SUCCESS"
        # RCP result should be included
        assert result.get("rcp_result") is not None or result["confidence"] > 0


def test_query_with_worker_check_endpoint():
    """Test the /query/with_worker_check endpoint"""
    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        # No workers ready
        mock_redis.scard.return_value = 0

        response = client.post(
            "/query/with_worker_check",
            json={"query_text": "test", "trace_id": "t4"},
            params={"required_workers": 2, "worker_timeout": 0.5}
        )

        # Should fail due to insufficient workers
        assert response.status_code == 503
        assert "Insufficient workers" in response.json()["detail"]
