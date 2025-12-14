"""
Baseline RAG Agent for DPR-RC Comparison

This is a simple RAG implementation WITHOUT consensus mechanisms.
Used as a baseline for comparing DPR-RC performance.

Key differences from DPR-RC:
- No peer voting (L3 RCP)
- No multi-agent consensus
- Single retrieval, single response
- Always returns confidence=1.0 (naive RAG)

Deployed as a separate Cloud Run service for fair cloud-to-cloud comparison.
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import chromadb

# Configuration
HISTORY_BUCKET = os.getenv("HISTORY_BUCKET", None)
DATA_SCALE = os.getenv("DATA_SCALE", "medium")
DATA_PREFIX = "synthetic_history/v2"


class QueryRequest(BaseModel):
    query_text: str
    trace_id: str = ""
    timestamp_context: Optional[str] = None


class QueryResponse(BaseModel):
    trace_id: str
    final_answer: str
    confidence: float
    status: str
    sources: List[str]
    latency_ms: float


class BaselineRAGWorker:
    """
    Simple RAG worker without consensus mechanisms.
    Retrieves documents and returns immediately - no voting, no verification.
    """

    def __init__(self):
        self.chroma = chromadb.Client()
        self._collections: Dict[str, Any] = {}
        self._initialize_data()

    def _initialize_data(self):
        """Initialize data from GCS or generate fallback."""
        if HISTORY_BUCKET:
            self._load_from_gcs()
        else:
            self._generate_fallback_data()

    def _load_from_gcs(self) -> bool:
        """Load data from GCS bucket."""
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(HISTORY_BUCKET)

            prefix = f"{DATA_PREFIX}/{DATA_SCALE}/shards/"
            blobs = list(bucket.list_blobs(prefix=prefix))

            if not blobs:
                print(f"No data found in gs://{HISTORY_BUCKET}/{prefix}, using fallback")
                self._generate_fallback_data()
                return False

            for blob in blobs:
                if blob.name.endswith('.json'):
                    filename = blob.name.split('/')[-1]
                    shard_id = filename.replace('shard_', '').replace('.json', '')

                    content = blob.download_as_text()
                    events = json.loads(content)

                    collection = self._get_collection(shard_id)
                    if collection.count() == 0:
                        self._ingest_events(events, collection)

            print(f"Loaded data from GCS: {HISTORY_BUCKET}")
            return True

        except Exception as e:
            print(f"GCS load failed: {e}, using fallback")
            self._generate_fallback_data()
            return False

    def _generate_fallback_data(self):
        """Generate minimal fallback data."""
        for year in range(2015, 2026):
            shard_id = str(year)
            collection = self._get_collection(shard_id)

            if collection.count() == 0:
                docs = []
                for i in range(20):
                    docs.append({
                        "id": f"baseline_{year}_{i}",
                        "content": f"Historical record from {year}. Research milestone {i} achieved. Progress in domain area.",
                        "metadata": {"year": year, "index": i}
                    })
                self._bulk_insert(docs, collection)

        # Broadcast shard
        broadcast = self._get_collection("broadcast")
        if broadcast.count() == 0:
            docs = [
                {"id": f"broadcast_{i}", "content": f"General knowledge record {i}. Cross-domain information.", "metadata": {}}
                for i in range(50)
            ]
            self._bulk_insert(docs, broadcast)

    def _get_collection(self, shard_id: str):
        """Get or create a collection."""
        if shard_id not in self._collections:
            self._collections[shard_id] = self.chroma.get_or_create_collection(
                name=f"baseline_shard_{shard_id}",
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[shard_id]

    def _ingest_events(self, events: List[Dict], collection):
        """Ingest events into a collection."""
        docs = [
            {
                "id": event.get('id', hashlib.md5(event['content'].encode()).hexdigest()[:12]),
                "content": event['content'],
                "metadata": {
                    "timestamp": event.get('timestamp', ''),
                    "topic": event.get('topic', ''),
                }
            }
            for event in events
        ]
        self._bulk_insert(docs, collection)

    def _bulk_insert(self, documents: List[Dict], collection):
        """Bulk insert documents."""
        if not documents:
            return

        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]

        try:
            collection.add(ids=ids, documents=contents, metadatas=metadatas)
        except Exception as e:
            if "duplicate" not in str(e).lower():
                print(f"Insert error: {e}")

    def query(self, query_text: str, timestamp_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Simple RAG query - retrieve and return.
        No consensus, no verification, no voting.
        """
        start = time.perf_counter()

        # Determine shard
        if timestamp_context:
            shard_id = timestamp_context[:4]
        else:
            shard_id = "broadcast"

        collection = self._get_collection(shard_id)

        # Simple retrieval
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=3
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if results['documents'] and results['documents'][0]:
                # Combine top results
                content = " ".join(results['documents'][0])
                return {
                    "answer": content,
                    "confidence": 1.0,  # Naive RAG is always confident
                    "sources": results['ids'][0],
                    "latency_ms": latency_ms,
                    "success": True
                }
            else:
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "sources": [],
                    "latency_ms": latency_ms,
                    "success": False
                }

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "answer": "",
                "confidence": 0.0,
                "sources": [],
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e)
            }


# Global worker instance
_worker: Optional[BaselineRAGWorker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker on startup."""
    global _worker
    print("Initializing Baseline RAG Worker...")
    _worker = BaselineRAGWorker()
    print("Baseline RAG Worker ready")
    yield
    print("Baseline RAG Worker shutting down")


app = FastAPI(title="DPR-Baseline-RAG", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "component": "baseline_rag",
        "description": "Simple RAG without consensus (for comparison)"
    }


@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """Handle baseline RAG query."""
    global _worker

    if _worker is None:
        _worker = BaselineRAGWorker()

    result = _worker.query(request.query_text, request.timestamp_context)

    return QueryResponse(
        trace_id=request.trace_id,
        final_answer=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
        status="SUCCESS" if result.get("success") else "FAILED",
        sources=result.get("sources", []),
        latency_ms=result.get("latency_ms", 0.0)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
