import pytest
import json
from unittest.mock import MagicMock, patch
from dpr_rc.passive_agent import PassiveWorker, redis_client, RFI_STREAM, EventType
from dpr_rc.models import ConsensusVote

@pytest.fixture
def mock_redis():
    with patch("dpr_rc.passive_agent.redis_client") as mock:
        yield mock

def test_passive_worker_processing(mock_redis):
    worker = PassiveWorker()
    
    # Mock retrieval to return a document
    worker.retrieve = MagicMock(return_value={
        "content": "Historic Event A occurred in 2020.",
        "id": "doc1",
        "metadata": {"year": 2020}
    })
    
    rfi_payload = {
        "trace_id": "trace_123",
        "query_text": "Event A?",
        "timestamp_context": "2020"
    }
    
    worker.process_rfi(rfi_payload)
    
    # Verify a vote was published
    mock_redis.publish.assert_called_once()
    args = mock_redis.publish.call_args[0]
    channel = args[0]
    message = json.loads(args[1])
    
    assert channel == "dpr:responses:trace_123"
    assert message["trace_id"] == "trace_123"
    assert message["content_snippet"] == "Historic Event A occurred in 2020."
    assert "confidence_score" in message

def test_passive_worker_no_result(mock_redis):
    worker = PassiveWorker()
    worker.retrieve = MagicMock(return_value=None)
    
    rfi_payload = {"trace_id": "t2", "query_text": "nothing"}
    worker.process_rfi(rfi_payload)
    
    # Verify NO vote was published
    mock_redis.publish.assert_not_called()
