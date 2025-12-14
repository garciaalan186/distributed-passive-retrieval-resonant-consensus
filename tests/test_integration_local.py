import pytest
import asyncio
import threading
import time
import json
import redis
from fastapi.testclient import TestClient
from unittest.mock import patch

# We need to ensure both active and passive agents use the SAME redis instance.
# Since they import `redis_client` globally, we must patch them both.

import dpr_rc.active_agent
import dpr_rc.passive_agent
import fakeredis

# Create a shared fakeredis client
fake_redis = fakeredis.FakeRedis(decode_responses=True)

@pytest.fixture(autouse=True)
def setup_redis():
    fake_redis.flushall()
    # Patch both modules to use the same fake redis
    with patch("dpr_rc.active_agent.redis_client", fake_redis), \
         patch("dpr_rc.passive_agent.redis_client", fake_redis):
        yield

def run_passive_worker_loop():
    # Helper to run the passive worker loop for a short time
    worker = dpr_rc.passive_agent.PassiveWorker()
    
    # Setup group
    try:
        fake_redis.xgroup_create(dpr_rc.passive_agent.RFI_STREAM, dpr_rc.passive_agent.GROUP_NAME, mkstream=True)
    except:
        pass

    # Process a few times then exit
    for _ in range(10): 
        streams = fake_redis.xreadgroup(dpr_rc.passive_agent.GROUP_NAME, "test_worker", {dpr_rc.passive_agent.RFI_STREAM: ">"}, count=1, block=100)
        if streams:
            for stream, messages in streams:
                for message_id, data in messages:
                    worker.process_rfi(data)
                    fake_redis.xack(dpr_rc.passive_agent.RFI_STREAM, dpr_rc.passive_agent.GROUP_NAME, message_id)
        time.sleep(0.1)

def test_full_integration_flow():
    # 1. Start Passive Worker in a thread
    worker_thread = threading.Thread(target=run_passive_worker_loop, daemon=True)
    worker_thread.start()

    # 2. Use Active Agent Client to send query
    client = TestClient(dpr_rc.active_agent.app)
    
    # We need to patch the Mock retrieval in PassiveWorker to ensure we get results
    # Ideally we'd ingest data, but for this test we'll patch the retrieve method dynamically if possible?
    # Or just rely on the default mock inside PassiveWorker (which returns item[0] if exists)
    
    # Let's ingest some mock data into Chroma if the PassiveWorker uses it?
    # The current PassiveWorker implementation has a `retrieve` method that queries chroma.
    # But for the test, `fakeredis` patches Redis, but `chromadb` is real (in-memory).
    
    # Let's patch `PassiveWorker.retrieve` to return a deterministic result for our query
    with patch("dpr_rc.passive_agent.PassiveWorker.retrieve") as mock_retrieve:
        mock_retrieve.return_value = {
            "content": "The answer to the ultimate question is 42.",
            "id": "doc_42",
            "metadata": {"source": "galaxy_guide"}
        }

        # Send Query
        response = client.post("/query", json={
            "query_text": "What is the answer?",
            "trace_id": "integration_test_1"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # 3. verify Result
        assert data["status"] == "SUCCESS", f"Failed with response: {data}"
        assert "42" in data["final_answer"]
        assert len(data["sources"]) > 0

    worker_thread.join(timeout=2)
