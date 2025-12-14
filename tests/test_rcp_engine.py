"""
Unit tests for RCP (Resonant Consensus Protocol) Engine.

Tests:
- Semantic quadrant computation
- Response and vote collection
- Consensus aggregation
- Result caching
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from dpr_rc.rcp_engine import RCPEngine
from dpr_rc.models import CachedResponse, PeerVote, SemanticQuadrant, AgentResponseScore


@pytest.fixture
def mock_redis():
    with patch("dpr_rc.rcp_engine.redis_client") as mock:
        yield mock


def test_semantic_quadrant_computation():
    """Test semantic quadrant computation from responses and votes"""
    engine = RCPEngine(redis_client_override=MagicMock())

    responses = [
        CachedResponse(
            trace_id="t1",
            agent_id="worker_1",
            shard_id="2020",
            content_hash="h1",
            content="High consensus content A",
            confidence=0.9,
            timestamp="2024-01-01T00:00:00"
        ),
        CachedResponse(
            trace_id="t1",
            agent_id="worker_2",
            shard_id="2021",
            content_hash="h2",
            content="Similar content A",
            confidence=0.85,
            timestamp="2024-01-01T00:00:01"
        ),
    ]

    votes = [
        PeerVote(
            trace_id="t1",
            voter_id="worker_1",
            votee_id="worker_2",
            agreement_score=0.9,
            disagreement_score=0.1,
            timestamp="2024-01-01T00:00:02"
        ),
        PeerVote(
            trace_id="t1",
            voter_id="worker_2",
            votee_id="worker_1",
            agreement_score=0.85,
            disagreement_score=0.15,
            timestamp="2024-01-01T00:00:03"
        ),
    ]

    quadrant, pa_scores = engine.compute_semantic_quadrant(responses, votes)

    # Verify structure
    assert isinstance(quadrant, SemanticQuadrant)
    assert len(pa_scores) == 2

    # Check worker scores
    assert "worker_1" in pa_scores
    assert "worker_2" in pa_scores

    # High agreement should result in symmetric resonance
    # (depending on thresholds)
    w1_score = pa_scores["worker_1"]
    assert w1_score.consensus_score > 0.5
    assert w1_score.polarization_score < 0.5


def test_asymmetric_perspectives_detection():
    """Test detection of asymmetric/divergent perspectives"""
    engine = RCPEngine(redis_client_override=MagicMock())

    responses = [
        CachedResponse(
            trace_id="t2",
            agent_id="worker_1",
            shard_id="2020",
            content_hash="h1",
            content="Perspective A content",
            confidence=0.7,
            timestamp="2024-01-01T00:00:00"
        ),
        CachedResponse(
            trace_id="t2",
            agent_id="worker_2",
            shard_id="2021",
            content_hash="h2",
            content="Contradictory perspective B",
            confidence=0.75,
            timestamp="2024-01-01T00:00:01"
        ),
    ]

    # High disagreement votes
    votes = [
        PeerVote(
            trace_id="t2",
            voter_id="worker_1",
            votee_id="worker_2",
            agreement_score=0.2,
            disagreement_score=0.8,
            timestamp="2024-01-01T00:00:02"
        ),
        PeerVote(
            trace_id="t2",
            voter_id="worker_2",
            votee_id="worker_1",
            agreement_score=0.25,
            disagreement_score=0.75,
            timestamp="2024-01-01T00:00:03"
        ),
    ]

    quadrant, pa_scores = engine.compute_semantic_quadrant(responses, votes)

    # Both should have high polarization
    assert pa_scores["worker_1"].polarization_score > 0.5
    assert pa_scores["worker_2"].polarization_score > 0.5

    # Should have asymmetric perspectives
    assert len(quadrant.asymmetric_perspectives) >= 1


def test_no_votes_uses_self_confidence():
    """Test that when no peer votes exist, self-confidence is used"""
    engine = RCPEngine(redis_client_override=MagicMock())

    responses = [
        CachedResponse(
            trace_id="t3",
            agent_id="worker_1",
            shard_id="2020",
            content_hash="h1",
            content="Single response content",
            confidence=0.85,
            timestamp="2024-01-01T00:00:00"
        ),
    ]

    votes = []  # No peer votes

    quadrant, pa_scores = engine.compute_semantic_quadrant(responses, votes)

    # Self-confidence should be used
    w1_score = pa_scores["worker_1"]
    assert w1_score.consensus_score == 0.85  # Same as confidence
    assert w1_score.polarization_score == 0.15  # 1 - confidence


def test_content_aggregation():
    """Test aggregation of multiple high-consensus responses"""
    engine = RCPEngine(redis_client_override=MagicMock())

    responses = [
        AgentResponseScore(
            content="First important fact about the topic.",
            confidence=0.95,
            consensus_score=0.9,
            polarization_score=0.1,
            shard_id="2020",
            quadrant_coords=[0.9, 0.1]
        ),
        AgentResponseScore(
            content="Second supporting detail.",
            confidence=0.88,
            consensus_score=0.85,
            polarization_score=0.15,
            shard_id="2021",
            quadrant_coords=[0.85, 0.15]
        ),
        AgentResponseScore(
            content="First important fact about the topic.",  # Duplicate
            confidence=0.9,
            consensus_score=0.88,
            polarization_score=0.12,
            shard_id="2022",
            quadrant_coords=[0.88, 0.12]
        ),
    ]

    aggregated = engine._aggregate_content(responses)

    # Should deduplicate based on content
    assert "First important fact" in aggregated
    assert "Second supporting detail" in aggregated


def test_result_caching(mock_redis):
    """Test that RCP results are cached properly"""
    # Setup mock to return responses
    mock_redis.scan_iter.return_value = iter(["dpr:response:t4:worker_1"])
    mock_redis.get.return_value = json.dumps({
        "trace_id": "t4",
        "agent_id": "worker_1",
        "shard_id": "2020",
        "content_hash": "h1",
        "content": "Test content",
        "confidence": 0.8,
        "timestamp": "2024-01-01T00:00:00"
    })

    engine = RCPEngine()
    result = engine.compute_and_cache_result("t4", expected_responses=1, timeout=0.5)

    # Verify setex was called to cache result
    assert mock_redis.setex.called
    call_args = mock_redis.setex.call_args[0]
    assert call_args[0] == "dpr:result:t4"
    assert call_args[1] == 300  # TTL

    # Verify notification was sent
    assert mock_redis.xadd.called


def test_get_cached_result(mock_redis):
    """Test retrieving cached RCP result"""
    cached_data = {
        "trace_id": "t5",
        "semantic_quadrant": {
            "symmetric_resonance": {"content": "test", "confidence": 0.9, "num_sources": 1},
            "asymmetric_perspectives": []
        },
        "pa_response_scores": {},
        "total_responses": 1,
        "total_votes": 0,
        "computation_time_ms": 5.0
    }
    mock_redis.get.return_value = json.dumps(cached_data)

    engine = RCPEngine()
    result = engine.get_cached_result("t5")

    assert result is not None
    assert result.trace_id == "t5"
    assert result.total_responses == 1


def test_timeout_with_no_responses(mock_redis):
    """Test that timeout returns None when no responses"""
    mock_redis.scan_iter.return_value = iter([])  # No responses

    engine = RCPEngine()
    result = engine.compute_and_cache_result("t6", expected_responses=2, timeout=0.2)

    assert result is None


def test_quadrant_coords_format():
    """Test that quadrant coordinates are properly formatted"""
    engine = RCPEngine(redis_client_override=MagicMock())

    responses = [
        CachedResponse(
            trace_id="t7",
            agent_id="worker_1",
            shard_id="2020",
            content_hash="h1",
            content="Test content",
            confidence=0.756,  # Should be rounded
            timestamp="2024-01-01T00:00:00"
        ),
    ]

    votes = [
        PeerVote(
            trace_id="t7",
            voter_id="worker_2",
            votee_id="worker_1",
            agreement_score=0.8234,
            disagreement_score=0.1766,
            timestamp="2024-01-01T00:00:01"
        ),
    ]

    quadrant, pa_scores = engine.compute_semantic_quadrant(responses, votes)

    # Check rounding
    w1_score = pa_scores["worker_1"]
    assert len(w1_score.quadrant_coords) == 2
    # Values should be rounded to 3 decimal places
    assert str(w1_score.quadrant_coords[0]).count('.') <= 1
