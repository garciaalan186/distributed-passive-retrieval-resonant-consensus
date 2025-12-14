"""
DPR-RC Core Package

Distributed Passive Retrieval with Resonant Consensus

Architecture: Cache-Based Response Model
- Passive agents are shard-agnostic workers
- Responses cached in Redis (not Pub/Sub)
- RCP Engine computes semantic quadrant topology
- Worker readiness verified before benchmark execution
"""

__version__ = "0.2.0"

from .models import (
    ComponentType, EventType, QueryRequest, RetrievalResult,
    CachedResponse, PeerVote, RCPResult, SemanticQuadrant, AgentResponseScore
)
from .rcp_engine import RCPEngine, wait_for_rcp_result, trigger_rcp_computation
from .passive_agent import PassiveWorker, wait_for_workers, get_ready_workers
