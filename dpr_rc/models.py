from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import uuid

class ComponentType(str, Enum):
    ACTIVE_CONTROLLER = "ActiveAgentController"
    PASSIVE_WORKER = "PassiveAgentWorker"
    ROUTER = "Router"
    RCP_ENGINE = "RCPEngine"

class EventType(str, Enum):
    RFI_BROADCAST = "RFI_Broadcast"
    VOTE_CAST = "Vote_Cast"
    CONSENSUS_REACHED = "Consensus_Reached"
    HALLUCINATION_DETECTED = "Hallucination_Detected"
    SYSTEM_INIT = "System_Init"
    PEER_VOTE_CAST = "Peer_Vote_Cast"
    RESPONSE_CACHED = "Response_Cached"
    WORKER_READY = "Worker_Ready"

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

# Legacy ConsensusVote - kept for backward compatibility
class ConsensusVote(BaseModel):
    trace_id: str
    worker_id: str
    content_hash: str
    confidence_score: float
    semantic_quadrant: List[float] # [x, y] coordinates
    content_snippet: str


# ========== NEW CACHE-BASED ARCHITECTURE MODELS ==========

class CachedResponse(BaseModel):
    """
    Response cached by passive agent in Redis.
    Key pattern: dpr:response:{trace_id}:{agent_id}
    TTL: 60 seconds
    """
    trace_id: str
    agent_id: str
    shard_id: str
    content_hash: str
    content: Optional[str] = None
    confidence: float
    timestamp: str

class PeerVote(BaseModel):
    """
    Peer vote cast by one passive agent on another's response.
    Key pattern: dpr:vote:{trace_id}:{voter_id}:{votee_id}
    TTL: 60 seconds
    """
    trace_id: str
    voter_id: str
    votee_id: str
    agreement_score: float  # v+ component
    disagreement_score: float  # v- component
    timestamp: str

class AgentResponseScore(BaseModel):
    """Individual agent's response with RCP-computed scores."""
    content: str
    confidence: float
    consensus_score: float  # v+ aggregated from peer votes
    polarization_score: float  # v- aggregated from peer votes
    shard_id: str
    quadrant_coords: List[float]  # [v+, v-]

class SemanticQuadrant(BaseModel):
    """
    RCP-computed semantic quadrant result.
    Per Mathematical Model Section 6.2:
    Maps responses to ⟨v+, v−⟩ coordinates based on peer vote alignment.
    """
    symmetric_resonance: Dict[str, Any]  # High agreement, low polarization
    asymmetric_perspectives: List[Dict[str, Any]]  # Divergent but confident

class RCPResult(BaseModel):
    """
    Final result from Resonant Consensus Protocol.
    Key pattern: dpr:result:{trace_id}
    TTL: 300 seconds
    """
    trace_id: str
    semantic_quadrant: SemanticQuadrant
    pa_response_scores: Dict[str, AgentResponseScore]
    total_responses: int
    total_votes: int
    computation_time_ms: float

class RetrievalResult(BaseModel):
    trace_id: str
    final_answer: str
    confidence: float
    status: str
    sources: List[str]
    superposition: Optional[Dict[str, Any]] = None
    rcp_result: Optional[RCPResult] = None
