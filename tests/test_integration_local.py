"""
Integration tests for DPR-RC system (Cache-Based Architecture).

These tests verify the complete flow:
Active Agent -> Redis RFI -> Passive Worker -> Redis Cache -> RCP -> Active Agent
"""

import pytest
import asyncio
import threading
import time
import json
import redis
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import dpr_rc.active_agent
import dpr_rc.passive_agent
import dpr_rc.rcp_engine
import fakeredis


# Create a shared fakeredis client
fake_redis = fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture(autouse=True)
def setup_redis():
    """Setup fake redis for all tests"""
    fake_redis.flushall()
    # Patch all modules to use the same fake redis
    with patch("dpr_rc.active_agent.redis_client", fake_redis), \
         patch("dpr_rc.passive_agent.redis_client", fake_redis), \
         patch("dpr_rc.rcp_engine.redis_client", fake_redis):
        yield


def run_passive_worker_loop(worker, iterations=10):
    """Helper to run the passive worker loop for a controlled number of iterations"""
    # Setup consumer group
    try:
        fake_redis.xgroup_create(
            dpr_rc.passive_agent.RFI_STREAM,
            dpr_rc.passive_agent.GROUP_NAME,
            mkstream=True
        )
    except redis.exceptions.ResponseError:
        pass  # Group already exists

    # Process messages for a limited number of iterations
    for _ in range(iterations):
        streams = fake_redis.xreadgroup(
            dpr_rc.passive_agent.GROUP_NAME,
            "test_worker",
            {dpr_rc.passive_agent.RFI_STREAM: ">"},
            count=1,
            block=100
        )
        if streams:
            for stream, messages in streams:
                for message_id, data in messages:
                    worker.process_rfi(data)
                    fake_redis.xack(
                        dpr_rc.passive_agent.RFI_STREAM,
                        dpr_rc.passive_agent.GROUP_NAME,
                        message_id
                    )
        time.sleep(0.05)


def test_full_integration_flow():
    """Test complete flow from query to consensus using cache-based architecture"""
    # 1. Create and configure passive worker
    worker = dpr_rc.passive_agent.PassiveWorker()

    # Ingest test data into the broadcast shard
    test_events = [
        {
            "id": "doc_42",
            "content": "The answer to the ultimate question of life, the universe, and everything is 42.",
            "metadata": {"source": "galaxy_guide", "year": 2020}
        },
        {
            "id": "doc_meaning",
            "content": "The meaning of life has been debated by philosophers for centuries.",
            "metadata": {"source": "philosophy_101", "year": 2019}
        }
    ]
    worker.ingest_benchmark_data(test_events, shard_id="broadcast")

    # 2. Start Passive Worker in a thread
    worker_thread = threading.Thread(
        target=run_passive_worker_loop,
        args=(worker, 15),
        daemon=True
    )
    worker_thread.start()

    # Give worker time to start and register as ready
    time.sleep(0.3)

    # 3. Use Active Agent Client to send query
    client = TestClient(dpr_rc.active_agent.app)

    # Send Query
    response = client.post("/query", json={
        "query_text": "What is the answer to life?",
        "trace_id": "integration_test_1"
    })

    # Wait for worker thread to complete
    worker_thread.join(timeout=3)

    assert response.status_code == 200
    data = response.json()

    # Verify we got a response
    assert "trace_id" in data
    assert data["trace_id"] == "integration_test_1"


def test_passive_worker_shard_agnostic_retrieval():
    """Test that passive worker can retrieve from any shard dynamically"""
    worker = dpr_rc.passive_agent.PassiveWorker()

    # Ingest data into different shards
    worker.ingest_benchmark_data([
        {"id": "2019_doc", "content": "Data from 2019 about topic X."}
    ], shard_id="2019")

    worker.ingest_benchmark_data([
        {"id": "2020_doc", "content": "Data from 2020 about topic Y."}
    ], shard_id="2020")

    worker.ingest_benchmark_data([
        {"id": "2021_doc", "content": "Data from 2021 about topic Z."}
    ], shard_id="2021")

    # Retrieve from different shards
    doc_2019 = worker.retrieve_from_shard("2019", "topic X")
    doc_2020 = worker.retrieve_from_shard("2020", "topic Y")
    doc_2021 = worker.retrieve_from_shard("2021", "topic Z")

    assert doc_2019 is not None
    assert "2019" in doc_2019["content"]

    assert doc_2020 is not None
    assert "2020" in doc_2020["content"]

    assert doc_2021 is not None
    assert "2021" in doc_2021["content"]


def test_passive_worker_confidence_calculation():
    """Test the L2 verification confidence calculation"""
    worker = dpr_rc.passive_agent.PassiveWorker()

    # Test confidence calculation with matching content
    confidence = worker.verify_l2(
        content="The quantum computer achieved breakthrough results in 2023.",
        query="quantum computer breakthrough",
        depth=0
    )
    assert confidence > 0.3  # Should have reasonable confidence due to overlap

    # Test with non-matching content
    confidence_low = worker.verify_l2(
        content="Weather patterns in tropical regions.",
        query="quantum computer breakthrough",
        depth=0
    )
    assert confidence_low < confidence  # Should be lower


def test_response_caching():
    """Test that responses are properly cached in Redis"""
    worker = dpr_rc.passive_agent.PassiveWorker()

    # Cache a response
    worker.cache_response(
        trace_id="cache_test_1",
        shard_id="2020",
        content="Test cached content",
        confidence=0.85
    )

    # Verify it's in Redis
    keys = list(fake_redis.scan_iter(match="dpr:response:cache_test_1:*"))
    assert len(keys) == 1

    # Verify content
    data = json.loads(fake_redis.get(keys[0]))
    assert data["trace_id"] == "cache_test_1"
    assert data["shard_id"] == "2020"
    assert data["confidence"] == 0.85


def test_worker_readiness_registration():
    """Test that workers register as ready"""
    worker = dpr_rc.passive_agent.PassiveWorker()

    # Check that worker registered in ready set
    ready_count = fake_redis.scard(dpr_rc.passive_agent.WORKERS_READY_KEY)
    assert ready_count >= 1

    # Verify worker ID is in set
    is_member = fake_redis.sismember(
        dpr_rc.passive_agent.WORKERS_READY_KEY,
        worker.worker_id
    )
    assert is_member


def test_rcp_engine_consensus_computation():
    """Test RCP engine consensus computation"""
    from dpr_rc.models import CachedResponse, PeerVote
    from dpr_rc.rcp_engine import RCPEngine

    # Setup test responses in cache
    responses = [
        CachedResponse(
            trace_id="rcp_test",
            agent_id="worker_1",
            shard_id="2020",
            content_hash="h1",
            content="Consensus content A",
            confidence=0.9,
            timestamp="2024-01-01T00:00:00"
        ),
        CachedResponse(
            trace_id="rcp_test",
            agent_id="worker_2",
            shard_id="2021",
            content_hash="h2",
            content="Consensus content A similar",
            confidence=0.85,
            timestamp="2024-01-01T00:00:01"
        ),
    ]

    for resp in responses:
        key = f"dpr:response:rcp_test:{resp.agent_id}"
        fake_redis.setex(key, 60, resp.model_dump_json())

    # Setup peer votes
    votes = [
        PeerVote(
            trace_id="rcp_test",
            voter_id="worker_1",
            votee_id="worker_2",
            agreement_score=0.8,
            disagreement_score=0.2,
            timestamp="2024-01-01T00:00:02"
        ),
        PeerVote(
            trace_id="rcp_test",
            voter_id="worker_2",
            votee_id="worker_1",
            agreement_score=0.85,
            disagreement_score=0.15,
            timestamp="2024-01-01T00:00:03"
        ),
    ]

    for vote in votes:
        key = f"dpr:vote:rcp_test:{vote.voter_id}:{vote.votee_id}"
        fake_redis.setex(key, 60, vote.model_dump_json())

    # Create RCP engine and compute
    engine = RCPEngine(redis_client_override=fake_redis)
    result = engine.compute_and_cache_result("rcp_test", expected_responses=2, timeout=1.0)

    assert result is not None
    assert result.trace_id == "rcp_test"
    assert result.total_responses == 2
    assert result.total_votes == 2

    # Verify result was cached
    cached_result = fake_redis.get("dpr:result:rcp_test")
    assert cached_result is not None


def test_data_ingestion_to_shards():
    """Test that benchmark data can be ingested to specific shards"""
    worker = dpr_rc.passive_agent.PassiveWorker()

    # Ingest data to specific shard
    events = [
        {"id": f"shard_event_{i}", "content": f"Test shard event content {i}"}
        for i in range(10)
    ]
    worker.ingest_benchmark_data(events, shard_id="test_shard")

    # Verify data was added to correct shard
    collection = worker._get_shard_collection("test_shard")
    assert collection.count() >= 10


def test_peer_voting_flow():
    """Test that peer votes are cast and cached"""
    from dpr_rc.models import CachedResponse

    worker = dpr_rc.passive_agent.PassiveWorker()

    # Create a fake peer response in cache
    peer_response = CachedResponse(
        trace_id="vote_test",
        agent_id="peer_worker",
        shard_id="2020",
        content_hash="abc123",
        content="Peer response content about the topic.",
        confidence=0.8,
        timestamp="2024-01-01T00:00:00"
    )
    fake_redis.setex(
        f"dpr:response:vote_test:peer_worker",
        60,
        peer_response.model_dump_json()
    )

    # Worker casts votes
    my_content = "My response content about the topic."
    worker.cast_peer_votes("vote_test", my_content, wait_time=0.1)

    # Check if vote was cached
    time.sleep(0.2)
    vote_keys = list(fake_redis.scan_iter(match="dpr:vote:vote_test:*"))

    # Should have at least one vote
    assert len(vote_keys) >= 1


def test_health_check_endpoints():
    """Test health check endpoints"""
    client = TestClient(dpr_rc.active_agent.app)

    # Active agent health
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["architecture"] == "cache-based"

    # Workers ready endpoint
    response = client.get("/workers/ready")
    assert response.status_code == 200
    assert "ready_count" in response.json()
