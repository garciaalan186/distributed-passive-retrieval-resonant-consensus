#!/usr/bin/env python3
"""
Download complete exchange history for a DPR-RC query.

This script fetches all log entries for a given trace_id from Cloud Logging
and reconstructs the complete message exchange sequence across all services
(Active Controller, Passive Workers, SLM Service).

Usage:
    python scripts/download_query_history.py <trace_id> [--format json|markdown]
    python scripts/download_query_history.py <trace_id> --output query_trace.json
"""

import sys
import json
import subprocess
from typing import List, Dict, Any
from datetime import datetime


def fetch_logs_for_trace(trace_id: str, project: str = "geometric-mnemic-manifolds-bm") -> List[Dict]:
    """Fetch all log entries for a given trace_id from Cloud Logging"""
    filter_query = f'jsonPayload.trace_id="{trace_id}"'

    cmd = [
        "gcloud", "logging", "read", filter_query,
        "--project", project,
        "--format", "json",
        "--order", "asc",  # Chronological order
        "--limit", "1000"  # Safety limit
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logs = json.loads(result.stdout) if result.stdout else []
        return logs
    except subprocess.CalledProcessError as e:
        print(f"Error fetching logs: {e.stderr}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing log JSON: {e}", file=sys.stderr)
        return []


def extract_message_exchanges(log_entries: List[Dict]) -> List[Dict]:
    """Extract structured message exchanges from log entries"""
    exchanges = []

    for entry in log_entries:
        payload = entry.get("jsonPayload", {})
        message_type = payload.get("message_type")

        # Only include actual message exchanges, not internal events
        relevant_types = [
            "client_query", "client_response",
            "slm_enhance_query", "slm_enhance_query_request", "slm_enhance_query_response",
            "slm_verify", "slm_verify_request", "slm_verify_response",
            "slm_hallucination_check_request", "slm_hallucination_check_response",
            "worker_rfi", "worker_votes", "vote_response",
            "rfi_received", "vote_created",
            "chromadb_query", "chromadb_results"
        ]

        if message_type in relevant_types:
            exchanges.append({
                "timestamp": entry.get("timestamp"),
                "component": payload.get("component", "unknown"),
                "direction": payload.get("direction"),
                "message_type": message_type,
                "request_payload": payload.get("request_payload"),
                "response_payload": payload.get("response_payload"),
                "internal_payload": payload.get("internal_payload"),
                "metadata": payload.get("metadata", {})
            })

    return exchanges


def format_as_markdown(trace_id: str, exchanges: List[Dict]) -> str:
    """Format exchange history as readable markdown"""
    md = f"# DPR-RC Query Exchange History\n\n"
    md += f"**Trace ID**: `{trace_id}`\n"
    md += f"**Total Exchanges**: {len(exchanges)}\n"
    md += f"**Generated**: {datetime.now().isoformat()}\n\n"
    md += "---\n\n"

    # Group by message flow
    for i, exchange in enumerate(exchanges, 1):
        md += f"## {i}. {exchange['message_type'].replace('_', ' ').title()}\n\n"
        md += f"- **Component**: {exchange['component']}\n"
        md += f"- **Direction**: {exchange['direction']}\n"
        md += f"- **Timestamp**: {exchange['timestamp']}\n\n"

        if exchange.get('request_payload'):
            md += "### Request\n```json\n"
            md += json.dumps(exchange['request_payload'], indent=2)
            md += "\n```\n\n"

        if exchange.get('response_payload'):
            md += "### Response\n```json\n"
            md += json.dumps(exchange['response_payload'], indent=2)
            md += "\n```\n\n"

        if exchange.get('internal_payload'):
            md += "### Internal Data\n```json\n"
            md += json.dumps(exchange['internal_payload'], indent=2)
            md += "\n```\n\n"

        if exchange.get('metadata'):
            md += "**Metadata**:\n"
            for key, value in exchange['metadata'].items():
                md += f"- {key}: {value}\n"
            md += "\n"

        md += "---\n\n"

    return md


def format_as_json(trace_id: str, exchanges: List[Dict]) -> str:
    """Format exchange history as JSON"""
    return json.dumps({
        "trace_id": trace_id,
        "fetched_at": datetime.now().isoformat(),
        "exchange_count": len(exchanges),
        "exchanges": exchanges
    }, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_query_history.py <trace_id> [--format json|markdown] [--output file]")
        print("\nExamples:")
        print("  python download_query_history.py abc-123-xyz")
        print("  python download_query_history.py abc-123-xyz --format json")
        print("  python download_query_history.py abc-123-xyz --output trace.md")
        sys.exit(1)

    trace_id = sys.argv[1]
    output_format = "markdown"
    output_file = None

    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--format" and i + 1 < len(sys.argv):
            output_format = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    print(f"Fetching logs for trace_id: {trace_id}...")
    log_entries = fetch_logs_for_trace(trace_id)
    print(f"Found {len(log_entries)} log entries")

    exchanges = extract_message_exchanges(log_entries)
    print(f"Extracted {len(exchanges)} message exchanges")

    if len(exchanges) == 0:
        print("\nNo message exchanges found for this trace_id.")
        print("Possible reasons:")
        print("  - Trace ID doesn't exist")
        print("  - Logs haven't been ingested yet (wait ~1 minute)")
        print("  - Enhanced logging not yet deployed")
        sys.exit(1)

    if output_format == "json":
        output = format_as_json(trace_id, exchanges)
    else:
        output = format_as_markdown(trace_id, exchanges)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"\n✓ Exchange history written to {output_file}")
    else:
        print("\n" + output)


if __name__ == "__main__":
    main()
