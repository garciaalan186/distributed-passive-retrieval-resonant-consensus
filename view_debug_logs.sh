#!/bin/bash
# view_debug_logs.sh
# View DPR-RC debug breakpoint logs from Cloud Run services

set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="${REGION:-us-central1}"
FRESHNESS="${FRESHNESS:-30m}"

echo "=== DPR-RC Debug Log Viewer ==="
echo "Project: $PROJECT_ID"
echo "Looking at logs from last: $FRESHNESS"
echo ""

# Check for --stream flag
if [ "$1" = "--stream" ] || [ "$1" = "-s" ]; then
    echo "📡 Streaming logs in real-time (Ctrl+C to stop)..."
    echo ""
    echo "Legend:"
    echo "  🔍 DEBUG BREAKPOINT - Step-by-step trace point"
    echo "  🚀 REQUEST START - New query received"
    echo "  ✅/❌ REQUEST END - Request completed"
    echo ""

    # Stream logs with filtering
    gcloud alpha logging tail \
        "resource.type=cloud_run_revision AND (resource.labels.service_name=dpr-active-controller OR resource.labels.service_name=dpr-passive-worker)" \
        --project="$PROJECT_ID" \
        --format='value(textPayload)' 2>/dev/null | \
        grep --line-buffered -E 'DEBUG|BREAKPOINT|REQUEST|🔍|🚀|✅|❌|📦|📍|🔗|📄|⏸️' || {
            echo "Error: Could not stream logs. You may need to enable the logging API."
            echo "Try: gcloud alpha logging tail --help"
        }
else
    echo "Fetching recent debug logs..."
    echo ""

    # Fetch recent logs with debug markers
    LOGS=$(gcloud logging read \
        "resource.type=cloud_run_revision AND (resource.labels.service_name=dpr-active-controller OR resource.labels.service_name=dpr-passive-worker) AND (textPayload:\"DEBUG\" OR textPayload:\"BREAKPOINT\" OR textPayload:\"REQUEST\")" \
        --project="$PROJECT_ID" \
        --format='value(timestamp, textPayload)' \
        --freshness="$FRESHNESS" \
        --limit=500 2>/dev/null || echo "")

    if [ -z "$LOGS" ]; then
        echo "No debug logs found in the last $FRESHNESS."
        echo ""
        echo "Make sure you ran the benchmark with DEBUG_BREAKPOINTS=true:"
        echo "  DEBUG_BREAKPOINTS=true ./run_cloud_benchmark.sh"
        echo ""
        echo "Or check if services are still deployed:"
        echo "  gcloud run services list --region=$REGION"
    else
        echo "$LOGS" | while IFS= read -r line; do
            # Add some formatting/highlighting
            echo "$line"
        done

        echo ""
        echo "--- End of Debug Logs ---"
    fi

    echo ""
    echo "Usage:"
    echo "  ./view_debug_logs.sh           # View recent logs"
    echo "  ./view_debug_logs.sh --stream  # Stream logs in real-time"
    echo "  FRESHNESS=1h ./view_debug_logs.sh  # View logs from last hour"
fi
