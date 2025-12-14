from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import uuid

class ComponentType(str, Enum):
    ACTIVE_CONTROLLER = "ActiveAgentController"
    PASSIVE_WORKER = "PassiveAgentWorker"
    ROUTER = "Router"

class EventType(str, Enum):
    RFI_BROADCAST = "RFI_Broadcast"
    VOTE_CAST = "Vote_Cast"
    CONSENSUS_REACHED = "Consensus_Reached"
    HALLUCINATION_DETECTED = "Hallucination_Detected"
    SYSTEM_INIT = "System_Init"

class LogEntry(BaseModel):
    trace_id: str
    timestamp: float = Field(default_factory=time.time)
    component: ComponentType
    event_type: EventType
    payload_hash: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None

class QueryRequest(BaseModel):
    query_text: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_context: Optional[str] = None  # e.g., "2023-01-01"

class ConsensusVote(BaseModel):
    trace_id: str
    worker_id: str
    content_hash: str
    confidence_score: float
    semantic_quadrant: List[float] # [x, y] coordinates
    content_snippet: str

class RetrievalResult(BaseModel):
    trace_id: str
    final_answer: str
    confidence: float
    status: str
    sources: List[str]
    superposition: Optional[Dict[str, Any]] = None


class ShardInfo(BaseModel):
    """Information about a loaded shard"""
    shard_id: str
    document_count: int
    embedding_model: Optional[str] = None
    loaded_from: str  # "gcs_embeddings", "gcs_raw", "redis_cache", "fallback"


class WorkerStatus(BaseModel):
    """Status information for a passive worker"""
    worker_id: str
    epoch: str
    mode: str  # "lazy_loading" or "eager_loading"
    loaded_shards: List[ShardInfo]
    bucket: Optional[str] = None
    scale: Optional[str] = None
    embedding_model: Optional[str] = None


class EmbeddingInfo(BaseModel):
    """Metadata about embeddings for a shard"""
    model_id: str
    dimension: int
    num_documents: int
    shard_id: str
    created_at: str
    checksum: str
