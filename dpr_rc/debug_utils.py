"""
Debug Utilities for DPR-RC Pipeline

Provides verbose step-by-step logging with optional pause points
for debugging the entire pipeline flow.

Enable with: DEBUG_BREAKPOINTS=true
Set pause duration with: DEBUG_PAUSE_SECONDS=2 (default: 2 seconds)
"""

import os
import sys
import json
import time
from typing import Any, Dict, Optional
from datetime import datetime

# Configuration
DEBUG_BREAKPOINTS = os.getenv("DEBUG_BREAKPOINTS", "false").lower() == "true"
DEBUG_PAUSE_SECONDS = float(os.getenv("DEBUG_PAUSE_SECONDS", "2"))
DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "false").lower() == "true"


def _truncate(value: Any, max_len: int = 500) -> str:
    """Truncate long values for display."""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + f"... [truncated, {len(s)} chars total]"
    return s


def _format_payload(payload: Any) -> str:
    """Format payload for debug output."""
    if payload is None:
        return "None"
    if isinstance(payload, dict):
        try:
            formatted = json.dumps(payload, indent=2, default=str)
            return _truncate(formatted, 2000)
        except Exception:
            return _truncate(str(payload), 2000)
    return _truncate(str(payload), 2000)


def debug_breakpoint(
    component: str,
    step: str,
    edge: str,
    payload: Any = None,
    extra_info: Optional[Dict[str, Any]] = None
):
    """
    Log a debug breakpoint with full payload information.

    Args:
        component: Component name (e.g., "ActiveController", "PassiveWorker")
        step: Step description (e.g., "Query Enhancement")
        edge: Edge traversal (e.g., "Controller → SLM")
        payload: The payload/data at this step
        extra_info: Additional debug information
    """
    if not DEBUG_BREAKPOINTS:
        return

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Create highly visible debug output
    separator = "=" * 80
    print(f"\n{separator}", file=sys.stderr, flush=True)
    print(f"🔍 DEBUG BREAKPOINT | {timestamp}", file=sys.stderr, flush=True)
    print(f"{separator}", file=sys.stderr, flush=True)
    print(f"📦 Component: {component}", file=sys.stderr, flush=True)
    print(f"📍 Step:      {step}", file=sys.stderr, flush=True)
    print(f"🔗 Edge:      {edge}", file=sys.stderr, flush=True)

    if extra_info:
        print(f"\n📊 Extra Info:", file=sys.stderr, flush=True)
        for key, value in extra_info.items():
            print(f"   {key}: {_truncate(value, 200)}", file=sys.stderr, flush=True)

    if payload is not None:
        print(f"\n📄 Payload:", file=sys.stderr, flush=True)
        formatted = _format_payload(payload)
        for line in formatted.split('\n'):
            print(f"   {line}", file=sys.stderr, flush=True)

    print(f"\n⏸️  PAUSED for {DEBUG_PAUSE_SECONDS}s (set DEBUG_PAUSE_SECONDS to adjust)",
          file=sys.stderr, flush=True)
    print(f"{separator}\n", file=sys.stderr, flush=True)

    # Pause to allow inspection
    time.sleep(DEBUG_PAUSE_SECONDS)


def debug_log(component: str, message: str, data: Any = None):
    """
    Log a debug message without pausing.

    Args:
        component: Component name
        message: Log message
        data: Optional data to log
    """
    if not (DEBUG_BREAKPOINTS or DEBUG_VERBOSE):
        return

    timestamp = datetime.utcnow().isoformat() + "Z"
    print(f"[DEBUG {timestamp}] [{component}] {message}", file=sys.stderr, flush=True)
    if data is not None:
        formatted = _format_payload(data)
        for line in formatted.split('\n')[:10]:  # Limit to first 10 lines
            print(f"    {line}", file=sys.stderr, flush=True)


def debug_start(component: str, trace_id: str):
    """Mark the start of a request processing."""
    if not DEBUG_BREAKPOINTS:
        return

    timestamp = datetime.utcnow().isoformat() + "Z"
    banner = "▶" * 40
    print(f"\n{banner}", file=sys.stderr, flush=True)
    print(f"🚀 REQUEST START | {component} | trace_id={trace_id}", file=sys.stderr, flush=True)
    print(f"   Timestamp: {timestamp}", file=sys.stderr, flush=True)
    print(f"{banner}\n", file=sys.stderr, flush=True)


def debug_end(component: str, trace_id: str, status: str, summary: Optional[Dict] = None):
    """Mark the end of a request processing."""
    if not DEBUG_BREAKPOINTS:
        return

    timestamp = datetime.utcnow().isoformat() + "Z"
    banner = "◀" * 40
    status_emoji = "✅" if status == "SUCCESS" else "❌"

    print(f"\n{banner}", file=sys.stderr, flush=True)
    print(f"{status_emoji} REQUEST END | {component} | trace_id={trace_id} | status={status}",
          file=sys.stderr, flush=True)
    print(f"   Timestamp: {timestamp}", file=sys.stderr, flush=True)

    if summary:
        print(f"   Summary:", file=sys.stderr, flush=True)
        for key, value in summary.items():
            print(f"      {key}: {_truncate(value, 100)}", file=sys.stderr, flush=True)

    print(f"{banner}\n", file=sys.stderr, flush=True)


# Edge labels for common traversals
class Edge:
    """Common edge labels for the DPR-RC architecture."""

    # Active Controller edges
    CLIENT_TO_CONTROLLER = "Client → Active Controller"
    CONTROLLER_TO_SLM_ENHANCE = "Controller → SLM (Query Enhancement)"
    CONTROLLER_TO_ROUTER = "Controller → L1 Router"
    CONTROLLER_TO_WORKERS_HTTP = "Controller → Workers (HTTP)"
    CONTROLLER_TO_WORKERS_REDIS = "Controller → Workers (Redis)"
    WORKERS_TO_CONTROLLER = "Workers → Controller (Votes)"
    CONTROLLER_CONSENSUS = "Controller: L3 Consensus"
    CONTROLLER_SUPERPOSITION = "Controller: Superposition Injection"
    CONTROLLER_TO_CLIENT = "Active Controller → Client"

    # Passive Worker edges
    CONTROLLER_TO_WORKER = "Controller → Worker (RFI)"
    WORKER_TO_GCS = "Worker → GCS (Shard Load)"
    WORKER_EMBEDDING = "Worker: Query Embedding"
    WORKER_RETRIEVAL = "Worker → ChromaDB (Retrieval)"
    WORKER_TO_SLM_VERIFY = "Worker → SLM (L2 Verification)"
    WORKER_L3_QUADRANT = "Worker: L3 Quadrant Calculation"
    WORKER_VOTE_CREATE = "Worker: Vote Creation"
    WORKER_TO_CONTROLLER = "Worker → Controller (Vote)"


# Convenience functions for common debug points
def debug_query_received(trace_id: str, query: str, timestamp_context: Optional[str] = None):
    """Debug point: Query received at Active Controller."""
    debug_start("ActiveController", trace_id)
    debug_breakpoint(
        component="ActiveController",
        step="1. Query Received",
        edge=Edge.CLIENT_TO_CONTROLLER,
        payload={
            "trace_id": trace_id,
            "query_text": query,
            "timestamp_context": timestamp_context
        }
    )


def debug_query_enhancement(trace_id: str, original: str, enhanced: str,
                           expansions: list, used: bool, time_ms: float = 0):
    """Debug point: Query enhancement via SLM."""
    debug_breakpoint(
        component="ActiveController",
        step="2. Query Enhancement via SLM",
        edge=Edge.CONTROLLER_TO_SLM_ENHANCE,
        payload={
            "original_query": original,
            "enhanced_query": enhanced,
            "expansions": expansions,
            "enhancement_used": used,
            "inference_time_ms": time_ms
        },
        extra_info={
            "SLM Service": os.getenv("SLM_SERVICE_URL", "not set"),
            "Enhancement Enabled": os.getenv("ENABLE_QUERY_ENHANCEMENT", "true")
        }
    )


def debug_routing(trace_id: str, query: str, target_shards: list):
    """Debug point: L1 routing decision."""
    debug_breakpoint(
        component="ActiveController",
        step="3. L1 Time-Sharded Routing",
        edge=Edge.CONTROLLER_TO_ROUTER,
        payload={
            "query": query,
            "target_shards": target_shards,
            "routing_strategy": "timestamp" if len(target_shards) == 1 else "broadcast"
        }
    )


def debug_http_worker_call(trace_id: str, worker_url: str, request_payload: dict):
    """Debug point: Calling passive workers via HTTP."""
    debug_breakpoint(
        component="ActiveController",
        step="4. HTTP Worker Call",
        edge=Edge.CONTROLLER_TO_WORKERS_HTTP,
        payload=request_payload,
        extra_info={
            "Worker URL": worker_url,
            "Timeout": os.getenv("HTTP_WORKER_TIMEOUT", "30.0")
        }
    )


def debug_http_worker_response(trace_id: str, worker_url: str, votes_count: int,
                               response_data: dict):
    """Debug point: Response received from HTTP worker."""
    debug_breakpoint(
        component="ActiveController",
        step="5. HTTP Worker Response",
        edge=Edge.WORKERS_TO_CONTROLLER,
        payload=response_data,
        extra_info={
            "Worker URL": worker_url,
            "Votes Received": votes_count
        }
    )


def debug_consensus_calculation(trace_id: str, votes_count: int, unique_candidates: int,
                                consensus_set: list, perspectival_set: list):
    """Debug point: L3 consensus calculation."""
    debug_breakpoint(
        component="ActiveController",
        step="6. L3 Resonant Consensus",
        edge=Edge.CONTROLLER_CONSENSUS,
        payload={
            "total_votes": votes_count,
            "unique_candidates": unique_candidates,
            "consensus_facts": consensus_set,
            "perspectival_claims": [p.get("claim", "")[:100] for p in perspectival_set]
        }
    )


def debug_final_response(trace_id: str, status: str, confidence: float,
                        answer: str, sources: list):
    """Debug point: Final response being sent to client."""
    debug_breakpoint(
        component="ActiveController",
        step="7. Final Response",
        edge=Edge.CONTROLLER_TO_CLIENT,
        payload={
            "status": status,
            "confidence": confidence,
            "final_answer": answer[:500],
            "sources": sources
        }
    )
    debug_end("ActiveController", trace_id, status, {
        "confidence": confidence,
        "answer_length": len(answer),
        "num_sources": len(sources)
    })


# Passive Worker debug points
def debug_rfi_received(trace_id: str, worker_id: str, rfi_data: dict):
    """Debug point: RFI received at passive worker."""
    debug_start("PassiveWorker", trace_id)
    debug_breakpoint(
        component=f"PassiveWorker[{worker_id}]",
        step="1. RFI Received",
        edge=Edge.CONTROLLER_TO_WORKER,
        payload=rfi_data
    )


def debug_shard_loading(worker_id: str, shard_id: str, strategy: str,
                        doc_count: int = 0, success: bool = True):
    """Debug point: Shard loading."""
    debug_breakpoint(
        component=f"PassiveWorker[{worker_id}]",
        step="2. Shard Loading",
        edge=Edge.WORKER_TO_GCS,
        payload={
            "shard_id": shard_id,
            "loading_strategy": strategy,
            "documents_loaded": doc_count,
            "success": success
        },
        extra_info={
            "Bucket": os.getenv("HISTORY_BUCKET", "not set"),
            "Scale": os.getenv("HISTORY_SCALE", "medium"),
            "Model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        }
    )


def debug_retrieval(worker_id: str, shard_id: str, query: str,
                   result: Optional[dict], distance: float = 0.0):
    """Debug point: Document retrieval from ChromaDB."""
    debug_breakpoint(
        component=f"PassiveWorker[{worker_id}]",
        step="3. Document Retrieval",
        edge=Edge.WORKER_RETRIEVAL,
        payload={
            "shard_id": shard_id,
            "query": query[:200],
            "result_found": result is not None,
            "result_content": result.get("content", "")[:300] if result else None,
            "distance": distance
        }
    )


def debug_l2_verification(worker_id: str, query: str, content: str,
                          confidence: float, method: str, slm_response: dict = None):
    """Debug point: L2 verification via SLM or fallback."""
    debug_breakpoint(
        component=f"PassiveWorker[{worker_id}]",
        step="4. L2 Verification",
        edge=Edge.WORKER_TO_SLM_VERIFY,
        payload={
            "query": query[:200],
            "content": content[:200],
            "verification_method": method,
            "confidence_score": confidence,
            "slm_response": slm_response
        },
        extra_info={
            "SLM Service": os.getenv("SLM_SERVICE_URL", "not set"),
            "Threshold": "0.3"
        }
    )


def debug_quadrant_calculation(worker_id: str, content_hash: str,
                                confidence: float, quadrant: list):
    """Debug point: L3 semantic quadrant calculation."""
    debug_breakpoint(
        component=f"PassiveWorker[{worker_id}]",
        step="5. L3 Quadrant Calculation",
        edge=Edge.WORKER_L3_QUADRANT,
        payload={
            "content_hash": content_hash,
            "confidence": confidence,
            "semantic_quadrant": quadrant,
            "quadrant_x": quadrant[0] if quadrant else None,
            "quadrant_y": quadrant[1] if len(quadrant) > 1 else None
        }
    )


def debug_vote_created(worker_id: str, trace_id: str, vote: dict):
    """Debug point: Vote creation."""
    debug_breakpoint(
        component=f"PassiveWorker[{worker_id}]",
        step="6. Vote Created",
        edge=Edge.WORKER_TO_CONTROLLER,
        payload=vote
    )
    debug_end(f"PassiveWorker[{worker_id}]", trace_id, "SUCCESS", {
        "confidence": vote.get("confidence_score", 0),
        "content_hash": vote.get("content_hash", "")[:12]
    })


def debug_worker_no_results(worker_id: str, trace_id: str, shard_id: str, reason: str):
    """Debug point: Worker found no relevant results."""
    debug_log(
        f"PassiveWorker[{worker_id}]",
        f"No results for shard {shard_id}: {reason}"
    )
    debug_end(f"PassiveWorker[{worker_id}]", trace_id, "NO_RESULTS", {
        "shard": shard_id,
        "reason": reason
    })
