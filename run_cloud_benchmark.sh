#!/bin/bash
# run_cloud_benchmark.sh
# End-to-end test: Build -> Deploy -> Benchmark
#
# Features:
# - Generates synthetic data once and stores in GCS
# - Deploys DPR-RC (active + passive workers) and Baseline RAG
# - Runs benchmark comparing cloud DPR-RC vs cloud Baseline

set -e

RESULTS_DIR="benchmark_results_cloud"
mkdir -p $RESULTS_DIR

echo "=== DPR-RC Cloud Benchmark Start ==="
PROJECT_ID=$(gcloud config get-value project)
echo "Project: $PROJECT_ID"

REPO_NAME="dpr-repo"
REGION="us-central1"
IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"
HISTORY_BUCKET="dpr-history-data-${PROJECT_ID}"

# 1. Infrastructure Setup
echo "--- Step 1: Setting Up Infrastructure ---"
./infrastructure.sh

# 2. Generate and Upload Synthetic Data (if not exists)
echo "--- Step 2: Checking Synthetic Data ---"
if gsutil -q stat "gs://${HISTORY_BUCKET}/synthetic_history/v2/medium/dataset.json" 2>/dev/null; then
    echo "Synthetic data already exists in GCS, skipping generation"
else
    echo "Generating and uploading synthetic data to GCS..."
    export HISTORY_BUCKET="${HISTORY_BUCKET}"
    python3 -m benchmark.generate_and_upload --scale medium
fi

# 3. Build Container
echo "--- Step 3: Building Container ---"
gcloud builds submit --tag $IMAGE_URI .

# 4. Deploy Services
echo "--- Step 4: Deploying Services ---"
chmod +x deploy_commands.sh
./deploy_commands.sh

# 5. Get Service URLs
echo "--- Step 5: Resolving Endpoints ---"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)')
BASELINE_URL=$(gcloud run services describe dpr-baseline-rag --region=$REGION --format='value(status.url)')

echo "DPR-RC Controller URL: $CONTROLLER_URL"
echo "Baseline RAG URL: $BASELINE_URL"

# Wait for services to be ready
echo "Waiting for services to initialize..."
sleep 10

# 6. Verify Services
echo "--- Step 6: Verifying Services ---"
echo "Checking DPR-RC..."
curl -s "${CONTROLLER_URL}/health" | head -c 200
echo ""
echo "Checking Baseline..."
curl -s "${BASELINE_URL}/health" | head -c 200
echo ""

# 7. Run Benchmark
echo "--- Step 7: Running Benchmark ---"
echo "DPR-RC: ${CONTROLLER_URL}"
echo "Baseline: ${BASELINE_URL}"

export CONTROLLER_URL="${CONTROLLER_URL}"
export BASELINE_URL="${BASELINE_URL}"
python3 -m benchmark.research_benchmark

# 8. Report
echo "--- Step 8: Results ---"
echo "Results available in: benchmark_results_research/"
echo ""
if [ -f "benchmark_results_research/RESEARCH_REPORT.md" ]; then
    echo "=== RESEARCH REPORT SUMMARY ==="
    head -50 benchmark_results_research/RESEARCH_REPORT.md
fi
