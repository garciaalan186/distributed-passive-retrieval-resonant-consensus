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
    cluster_id: str  # RCP v4: Which adversarial cluster this worker belongs to (e.g., "C_RECENT", "C_OLDER")
    content_hash: str
    confidence_score: float  # Continuous score for internal use
    binary_vote: int  # RCP v4: Binary vote {0, 1} based on threshold θ
    resonance_vector: List[float]  # RCP v4: [v+, v-] = approval rates from each cluster
    content_snippet: str
    author_cluster: Optional[str] = None  # RCP v4: Cluster of the artifact author
    document_ids: Optional[List[str]] = None  # Source document IDs for provenance tracking

class RetrievalResult(BaseModel):
    trace_id: str
    final_answer: Optional[str] = None  # A* interprets superposition
    confidence: Optional[float] = None  # A* determines confidence
    status: str
    sources: List[str]
    resonance_matrix: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = {"exclude_none": False}


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


class RCPConfig(BaseModel):
    """Resonant Consensus Protocol v4 Configuration"""
    theta: float = 0.5  # Cluster approval threshold (Eq. 1): fraction of agents in cluster needed to approve
    tau: float = 0.6    # Consensus threshold (Eq. 4): fraction of clusters needed for consensus
    vote_threshold: float = 0.5  # Confidence threshold for binary voting: confidence >= threshold → vote = 1

    # Temporal cluster definitions for DPR-RC
    # C_RECENT: shards from 2020 onwards (newer historical versions)
    # C_OLDER: shards before 2020 (older historical versions)
    recent_epoch_start: str = "2020-01-01"
