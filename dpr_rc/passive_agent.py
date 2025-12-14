import os
import time
import json
import redis
import threading
import torch
import numpy as np
import chromadb
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from .models import (
    ComponentType, EventType, LogEntry, ConsensusVote
)
from .logging_utils import StructuredLogger

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
WORKER_ID = os.getenv("HOSTNAME", f"worker-{os.getpid()}")
HISTORY_BUCKET = os.getenv("HISTORY_BUCKET", None)

logger = StructuredLogger(ComponentType.PASSIVE_WORKER)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

RFI_STREAM = "dpr:rfi"
GROUP_NAME = "passive_workers"
CONSUMER_NAME = WORKER_ID

# Mock SLM / Index
class PassiveWorker:
    def __init__(self):
        self.chroma = chromadb.Client() # In-memory for now, or connect to persistent
        self.collection = self.chroma.get_or_create_collection(name="history")
        self.setup_mock_data()
    
    def setup_mock_data(self):
        # In a real run, this loads from GCS bucket (HISTORY_BUCKET)
        # For the prototype, we assume the benchmark ingests data.
        pass

    def retrieve(self, query_text: str, timestamp_context: str) -> Dict[str, Any]:
        # Perform retrieval
        results = self.collection.query(
            query_texts=[query_text],
            n_results=1
        )
        if not results['documents'][0]:
            return None
            
        return {
            "content": results['documents'][0][0],
            "id": results['ids'][0][0],
            "metadata": results['metadatas'][0][0]
        }

    def verify_l2(self, content: str, query: str) -> float:
        # L2 Verification using quantized SLM
        # Here we mock it or use a tiny transformer if available.
        # "Phi-3" placeholder
        # Return confidence score 0.0 - 1.0
        # Logic: If query keywords in content, high score.
        score = 0.5
        if any(word in content.lower() for word in query.lower().split()):
            score += 0.4
        return score

    def calculate_quadrant(self, content: str) -> list:
        # L3 consensus quadrant calculation
        # Deterministic projection of content hash to 2D plane
        h = hash(content)
        x = (h % 100) / 100.0
        y = ((h >> 8) % 100) / 100.0
        return [x, y]

    def process_rfi(self, rfi_data: Dict[str, Any]):
        trace_id = rfi_data['trace_id']
        query_text = rfi_data['query_text']
        ts_context = rfi_data.get('timestamp_context')
        target_shards = rfi_data.get('target_shards')

        # Check if I am in the target shard (Mock: accept all or random)
        # In real impl, check year vs WORKER_YEAR env var.
        
        # 1. Retrieval
        doc = self.retrieve(query_text, ts_context)
        if not doc:
            return # No relevant history

        # 2. Verification (L2)
        confidence = self.verify_l2(doc['content'], query_text)
        
        # 3. Vote (L3)
        quadrant = self.calculate_quadrant(doc['content'])
        
        vote = ConsensusVote(
            trace_id=trace_id,
            worker_id=WORKER_ID,
            content_hash=logger.hash_payload(doc['content']),
            confidence_score=confidence,
            semantic_quadrant=quadrant,
            content_snippet=doc['content']
        )
        
        # Publish Vote to Stream logic (Active Controller reads this? No, it uses PubSub for speed)
        # Architecture says "Consensus Voting".
        # We publish to the response channel for the Controller to pick up.
        
        redis_client.publish(f"dpr:responses:{trace_id}", json.dumps(vote.model_dump()))
        
        logger.log_event(trace_id, EventType.VOTE_CAST, vote.model_dump(), metrics={"confidence": confidence})

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

# ... imports ...

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "healthy", "worker_id": WORKER_ID}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start worker thread
    worker_thread = threading.Thread(target=run_worker_loop, daemon=True)
    worker_thread.start()
    yield
    # Shutdown

app = FastAPI(lifespan=lifespan)

def run_worker_loop():
    worker = PassiveWorker()
    
    # Initialize Stream Group
    try:
        redis_client.xgroup_create(RFI_STREAM, GROUP_NAME, mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    logger.logger.info(f"Passive Worker {WORKER_ID} Listening...")
    
    while True:
        try:
            # Read new RFIs
            # using '>' to read only new messages
            streams = redis_client.xreadgroup(GROUP_NAME, CONSUMER_NAME, {RFI_STREAM: ">"}, count=1, block=2000)
            
            if streams:
                for stream, messages in streams:
                    for message_id, data in messages:
                        try:
                            worker.process_rfi(data)
                            # Acknowledge
                            redis_client.xack(RFI_STREAM, GROUP_NAME, message_id)
                        except Exception as e:
                            logger.logger.error(f"Error processing RFI: {e}")
                            
        except Exception as e:
            logger.logger.error(f"Worker Loop Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    # Local run
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
