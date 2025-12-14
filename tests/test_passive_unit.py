"""
Unit tests for Passive Agent Worker (Cache-Based Architecture)

Tests:
- Shard-agnostic operation
- Response caching (not Pub/Sub)
- Peer voting
- Worker readiness registration
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from dpr_rc.passive_agent import PassiveWorker, WORKERS_READY_KEY
from dpr_rc.models import CachedResponse, PeerVote


@pytest.fixture
def mock_redis():
    with patch("dpr_rc.passive_agent.redis_client") as mock:
        # Setup mock for SADD (worker registration)
        mock.sadd.return_value = 1
        mock.expire.return_value = True
        mock.sismember.return_value = True
        mock.scan_iter.return_value = iter([])
        yield mock


def test_passive_worker_shard_agnostic_initialization(mock_redis):
    """Test that passive worker initializes as shard-agnostic"""
    worker = PassiveWorker()

    # Worker should be shard-agnostic (no fixed epoch)
    assert hasattr(worker, 'worker_id')
    assert hasattr(worker, '_shard_collections')

    # Should have registered as ready
    mock_redis.sadd.assert_called()


def test_passive_worker_caches_response(mock_redis):
    """Test that worker caches response in Redis instead of Pub/Sub"""
    worker = PassiveWorker()

    # Mock retrieval to return a document
    worker.retrieve_from_shard = MagicMock(return_value={
        "content": "Historic Event A occurred in 2020.",
        "id": "doc1",
        "metadata": {"year": 2020}
    })

    rfi_payload = {
        "trace_id": "trace_123",
        "query_text": "Event A?",
        "target_shards": '["shard_2020"]',
        "timestamp_context": "2020"
    }

    worker.process_rfi(rfi_payload)

    # Verify response was cached (setex called), NOT published
    mock_redis.setex.assert_called()
    call_args = mock_redis.setex.call_args[0]
    assert call_args[0].startswith("dpr:response:trace_123:")

    # Verify publish was NOT called (old Pub/Sub approach)
    mock_redis.publish.assert_not_called()


def test_passive_worker_dynamic_shard_connection(mock_redis):
    """Test that worker can connect to any shard dynamically"""
    worker = PassiveWorker()

    # Connect to multiple shards
    collection_2019 = worker._get_shard_collection("2019")
    collection_2020 = worker._get_shard_collection("2020")
    collection_2021 = worker._get_shard_collection("2021")

    # All should be in the cache
    assert "2019" in worker._shard_collections
    assert "2020" in worker._shard_collections
    assert "2021" in worker._shard_collections

    # Should be different collections
    assert collection_2019.name != collection_2020.name


def test_passive_worker_no_result_no_cache(mock_redis):
    """Test that no response is cached when retrieval returns nothing"""
    worker = PassiveWorker()
    worker.retrieve_from_shard = MagicMock(return_value=None)

    rfi_payload = {"trace_id": "t2", "query_text": "nothing", "target_shards": '["broadcast"]'}
    worker.process_rfi(rfi_payload)

    # Verify NO response was cached
    # setex may be called for other purposes, check specific key pattern
    for call in mock_redis.setex.call_args_list:
        key = call[0][0]
        assert not key.startswith("dpr:response:t2:")


def test_passive_worker_registers_ready(mock_redis):
    """Test that worker registers in ready set"""
    worker = PassiveWorker()

    # Should have called SADD with workers ready key
    calls = [call for call in mock_redis.sadd.call_args_list
             if call[0][0] == WORKERS_READY_KEY]
    assert len(calls) > 0


def test_peer_vote_computation(mock_redis):
    """Test peer vote computation"""
    worker = PassiveWorker()

    my_content = "This is my response about topic X."
    peer_response = CachedResponse(
        trace_id="trace_1",
        agent_id="peer_worker",
        shard_id="2020",
        content_hash="abc123",
        content="This is peer response about topic X.",
        confidence=0.8,
        timestamp="2024-01-01T00:00:00"
    )

    vote = worker.compute_peer_vote(my_content, peer_response)

    assert isinstance(vote, PeerVote)
    assert vote.voter_id == worker.worker_id
    assert vote.votee_id == "peer_worker"
    assert 0 <= vote.agreement_score <= 1
    assert 0 <= vote.disagreement_score <= 1
    # Agreement + disagreement should approximately equal 1
    assert abs((vote.agreement_score + vote.disagreement_score) - 1.0) < 0.01


def test_confidence_calculation():
    """Test L2 verification confidence calculation"""
    with patch("dpr_rc.passive_agent.redis_client") as mock_redis:
        mock_redis.sadd.return_value = 1
        mock_redis.expire.return_value = True

        worker = PassiveWorker()

        # Test with matching content
        confidence = worker.verify_l2(
            content="Historical research milestone achieved in 2020",
            query="research milestone 2020",
            depth=0
        )
        assert confidence > 0.3

        # Test depth penalty
        confidence_deep = worker.verify_l2(
            content="Historical research milestone achieved in 2020",
            query="research milestone 2020",
            depth=2
        )
        assert confidence_deep < confidence  # Deeper = lower confidence


def test_cache_response_format(mock_redis):
    """Test that cached response has correct format"""
    worker = PassiveWorker()

    worker.cache_response(
        trace_id="test_trace",
        shard_id="2020",
        content="Test content",
        confidence=0.85
    )

    # Get the cached data
    call_args = mock_redis.setex.call_args[0]
    key = call_args[0]
    ttl = call_args[1]
    data = json.loads(call_args[2])

    assert key.startswith("dpr:response:test_trace:")
    assert ttl == 60  # Default TTL
    assert data["trace_id"] == "test_trace"
    assert data["shard_id"] == "2020"
    assert data["confidence"] == 0.85
    assert "content" in data
    assert "timestamp" in data
