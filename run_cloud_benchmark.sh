#!/bin/bash
# run_cloud_benchmark.sh
# End-to-end test: Data Prep -> Build -> Deploy -> Benchmark

set -e

RESULTS_DIR="benchmark_results_cloud"
mkdir -p $RESULTS_DIR

# Configuration
BENCHMARK_SCALE="${BENCHMARK_SCALE:-medium}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
HISTORY_BUCKET="${HISTORY_BUCKET:-dpr-history-data}"

echo "=== DPR-RC Cloud Benchmark Start ==="
PROJECT_ID=$(gcloud config get-value project)
echo "Project: $PROJECT_ID"
echo "Scale: $BENCHMARK_SCALE"
echo "Embedding Model: $EMBEDDING_MODEL"
echo "History Bucket: $HISTORY_BUCKET"
echo ""

# Export for Python scripts
export HISTORY_BUCKET
export EMBEDDING_MODEL

# 0. Ensure Dependencies
echo "--- Step 0: Checking Dependencies ---"
pip install -q google-cloud-storage sentence-transformers numpy 2>/dev/null || true

# 1. Ensure Raw Data and Embeddings Exist
echo ""
echo "--- Step 1: Ensuring Data & Embeddings ---"
echo "Checking if raw data and pre-computed embeddings exist in GCS..."
echo ""
echo "NOTE: Embedding computation is parallelized across time shards."
echo "      Each shard is independent - no cross-shard context needed."
echo ""

# This will:
# - Skip if embeddings already exist
# - Download raw JSON and compute embeddings in parallel if missing
# - Fail if no raw data exists (need to run with --force first)
python3 -m benchmark.generate_and_upload \
    --ensure-embeddings \
    --scale "$BENCHMARK_SCALE" \
    --model "$EMBEDDING_MODEL"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Embeddings not available. Generating raw data and embeddings..."
    python3 -m benchmark.generate_and_upload \
        --scale "$BENCHMARK_SCALE" \
        --model "$EMBEDDING_MODEL" \
        --force
fi

echo ""
echo "✓ Data and embeddings ready"

# 2. Infrastructure & Build
echo ""
echo "--- Step 2: Infrastructure & Build ---"
REPO_NAME="dpr-repo"
REGION="us-central1"
IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"

# Ensure infrastructure exists
echo "Ensuring infrastructure..."
./infrastructure.sh

echo "Submitting build to Cloud Build..."
gcloud builds submit --tag $IMAGE_URI .

# 3. Deploy Services
echo ""
echo "--- Step 3: Deploying Services ---"
chmod +x deploy_commands.sh
./deploy_commands.sh

# 4. Get Controller URL
echo ""
echo "--- Step 4: Resolving Endpoint ---"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)')
echo "Controller URL: $CONTROLLER_URL"

# 5. Run Benchmark
echo ""
echo "--- Step 5: Running Benchmark ---"
echo "Targeting: ${CONTROLLER_URL}"
echo "Scale: ${BENCHMARK_SCALE}"
echo ""

# Export for benchmark script
export CONTROLLER_URL="${CONTROLLER_URL}"
export HISTORY_SCALE="${BENCHMARK_SCALE}"

python3 -m benchmark.research_benchmark

# 6. Report
echo ""
echo "--- Step 6: Results ---"
echo "Results available in $RESULTS_DIR/"
echo ""
echo "=== DPR-RC Cloud Benchmark Complete ==="
