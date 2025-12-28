#!/bin/bash
# download_results.sh
# Download latest benchmark results from GCS

set -e

RESULTS_DIR="${RESULTS_DIR:-benchmark_results_cloud}"
mkdir -p "$RESULTS_DIR"

# Get project and bucket
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project configured. Run: gcloud config set project YOUR_PROJECT"
    exit 1
fi

HISTORY_BUCKET="dpr-history-data-${PROJECT_ID}"

echo "=== Downloading DPR-RC Benchmark Results ==="
echo "Project: $PROJECT_ID"
echo "Bucket: gs://${HISTORY_BUCKET}"
echo "Local dir: $RESULTS_DIR"
echo ""

# Check if bucket exists
if ! gsutil ls "gs://${HISTORY_BUCKET}" >/dev/null 2>&1; then
    echo "Error: Bucket gs://${HISTORY_BUCKET} does not exist or is not accessible"
    exit 1
fi

# List available results
echo "--- Available Results in GCS ---"
gsutil ls "gs://${HISTORY_BUCKET}/benchmark_results/" 2>/dev/null || {
    echo "No benchmark_results/ directory found in bucket."
    echo ""
    echo "Available directories:"
    gsutil ls "gs://${HISTORY_BUCKET}/"
    exit 1
}

# Download results
echo ""
echo "--- Downloading Results ---"
gsutil -m cp -r "gs://${HISTORY_BUCKET}/benchmark_results/*" "$RESULTS_DIR/" 2>/dev/null && {
    echo ""
    echo "✓ Results downloaded to: $RESULTS_DIR/"
    echo ""
    echo "--- Downloaded Files ---"
    find "$RESULTS_DIR" -name "*.json" -o -name "*.md" | head -20
} || {
    echo "Warning: Could not download results."
    echo ""
    echo "Checking Cloud Build logs instead..."
    echo "Recent builds:"
    gcloud builds list --limit=5 --format="table(id,status,createTime,duration)"
    echo ""
    echo "To view logs: gcloud builds log BUILD_ID"
}

echo ""
echo "=== Download Complete ==="
