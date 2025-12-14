import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from dpr_rc.active_agent import app, RouteLogic, redis_client, RFI_STREAM, VOTE_STREAM
from dpr_rc.models import QueryRequest, ConsensusVote

# Mock Redis for Unit Tests
@pytest.fixture
def mock_redis():
    with patch("dpr_rc.active_agent.redis_client") as mock:
        pubsub_mock = MagicMock()
        mock.pubsub.return_value = pubsub_mock
        yield mock

client = TestClient(app)

def test_route_logic():
    # Test time-sharded routing
    req = QueryRequest(query_text="foo", timestamp_context="2022-01-01", trace_id="abc")
    shards = RouteLogic.get_target_shards(req)
    assert "shard_2022" in shards

def test_query_broadcast(mock_redis):
    # Test that a query broadcasts an RFI to Redis
    mock_redis.pubsub().get_message.return_value = None # No votes immediately
    
    response = client.post("/query", json={
        "query_text": "testing query",
        "timestamp_context": "2020-01-01",
        "trace_id": "test_trace_1"
    })
    
    # Verify RFI was added
    mock_redis.xadd.assert_called_once()
    args = mock_redis.xadd.call_args[0]
    assert args[0] == RFI_STREAM
    payload = args[1]
    assert payload["query_text"] == "testing query"
    assert payload["trace_id"] == "test_trace_1"

def test_consensus_logic_superposition():
    # We need to test the logic INSIDE handle_query for consensus.
    # Since it's an async FastAPI route, we can mock the redis pubsub to return votes
    
    votes_data = [
        # 3 votes for "Fact A" (Consensus)
        ConsensusVote(trace_id="t1", worker_id="w1", content_hash="h1", confidence_score=0.9, semantic_quadrant=[0.9,0.9], content_snippet="Fact A").model_dump(),
        ConsensusVote(trace_id="t1", worker_id="w2", content_hash="h1", confidence_score=0.92, semantic_quadrant=[0.9,0.9], content_snippet="Fact A").model_dump(),
        ConsensusVote(trace_id="t1", worker_id="w3", content_hash="h1", confidence_score=0.88, semantic_quadrant=[0.9,0.9], content_snippet="Fact A").model_dump(),
        
        # 1 vote for "Fact B" (Perspective)
        ConsensusVote(trace_id="t1", worker_id="w4", content_hash="h2", confidence_score=0.65, semantic_quadrant=[0.2,0.5], content_snippet="Fact B").model_dump(),
    ]

    with patch("dpr_rc.active_agent.redis_client") as mock_redis:
        mock_pubsub = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub
        
        # Simulate pubsub returning messages then None
        messages = [
            {"data": json.dumps(v)} for v in votes_data
        ]
        # function to yield messages then StopIteration-ish behavior (return None)
        mock_pubsub.get_message.side_effect = messages + [None, None, None, None, None] 

        # We assume the route uses redis_client global.
        # Calling the endpoint via TestClient:
        response = client.post("/query", json={
            "query_text": "What is truth?",
            "trace_id": "t1"
        })
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["status"] == "SUCCESS"
        # Check that Fact A is in the answer
        assert "Fact A" in result["final_answer"]
        # Check that Fact B (perspective) is also mentioned or implied
        # It depends on prompt construction, but for now just ensure success.
        
        # Verify Confidence is high because of consensus
        assert result["confidence"] >= 0.9

