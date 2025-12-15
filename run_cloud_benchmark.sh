#!/bin/bash
# run_cloud_benchmark.sh
# End-to-end test: Data Prep -> Build -> Deploy -> Benchmark

set -e

RESULTS_DIR="benchmark_results_cloud"
mkdir -p $RESULTS_DIR

# Configuration
BENCHMARK_SCALE="${BENCHMARK_SCALE:-medium}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
REGION="${REGION:-us-central1}"

echo "=== DPR-RC Cloud Benchmark Start ==="
PROJECT_ID=$(gcloud config get-value project)
echo "Project: $PROJECT_ID"
echo "Scale: $BENCHMARK_SCALE"
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Region: $REGION"
echo ""

# Set bucket name based on project ID for uniqueness
HISTORY_BUCKET="${HISTORY_BUCKET:-dpr-history-data-${PROJECT_ID}}"
echo "History Bucket: $HISTORY_BUCKET"
echo ""

# Export for Python scripts
export HISTORY_BUCKET
export EMBEDDING_MODEL
export PROJECT_ID

# 0. Ensure Dependencies
echo "--- Step 0: Checking Dependencies ---"
# Note: numpy<2 required for compatibility with PyTorch/sentence-transformers
pip install -q google-cloud-storage sentence-transformers "numpy<2" 2>/dev/null || true

# 1. Ensure GCS Bucket Exists
echo ""
echo "--- Step 1: Ensuring GCS Bucket ---"
if gsutil ls -b "gs://${HISTORY_BUCKET}" >/dev/null 2>&1; then
    echo "✓ Bucket gs://${HISTORY_BUCKET} exists"
else
    echo "Creating bucket gs://${HISTORY_BUCKET}..."
    gsutil mb -l "${REGION}" "gs://${HISTORY_BUCKET}"
    echo "✓ Bucket created"
fi

# 2. Ensure Raw Data and Embeddings Exist
echo ""
echo "--- Step 2: Ensuring Data & Embeddings ---"
echo "Checking if raw data and pre-computed embeddings exist in GCS..."
echo ""
echo "NOTE: Embedding computation is parallelized across time shards."
echo "      Each shard is independent - no cross-shard context needed."
echo ""

# First try to ensure embeddings exist (will skip if already present)
python3 -m benchmark.generate_and_upload \
    --ensure-embeddings \
    --scale "$BENCHMARK_SCALE" \
    --model "$EMBEDDING_MODEL" || {

    echo ""
    echo "Embeddings not found. Generating raw data and embeddings..."
    python3 -m benchmark.generate_and_upload \
        --scale "$BENCHMARK_SCALE" \
        --model "$EMBEDDING_MODEL" \
        --force
}

echo ""
echo "✓ Data and embeddings ready"

# 3. Infrastructure & Build
echo ""
echo "--- Step 3: Infrastructure & Build ---"
REPO_NAME="dpr-repo"
IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"

# Ensure infrastructure exists
echo "Ensuring infrastructure..."
./infrastructure.sh

echo "Submitting build to Cloud Build..."
gcloud builds submit --tag $IMAGE_URI .

# 4. Deploy Services
echo ""
echo "--- Step 4: Deploying Services ---"
chmod +x deploy_commands.sh
./deploy_commands.sh

# 5. Get Controller URL
echo ""
echo "--- Step 5: Resolving Endpoint ---"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)')
echo "Controller URL: $CONTROLLER_URL"

# 6. Run Benchmark
echo ""
echo "--- Step 6: Running Benchmark ---"
echo "Targeting: ${CONTROLLER_URL}"
echo "Scale: ${BENCHMARK_SCALE}"
echo ""

# Export for benchmark script
export CONTROLLER_URL="${CONTROLLER_URL}"
export HISTORY_SCALE="${BENCHMARK_SCALE}"

python3 -m benchmark.research_benchmark

# 7. Report
echo ""
echo "--- Step 7: Results ---"
echo "Results available in $RESULTS_DIR/"
echo ""
echo "=== DPR-RC Cloud Benchmark Complete ==="
