#!/bin/bash
# run_cloud_benchmark.sh
# End-to-end test: Data Prep -> Build -> Deploy -> Benchmark
#
# IMPORTANT: Embedding computation happens in CLOUD BUILD, not locally.
# Your local machine only generates lightweight raw JSON.

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
# Always use project-specific bucket to avoid conflicts
HISTORY_BUCKET="dpr-history-data-${PROJECT_ID}"
echo "History Bucket: $HISTORY_BUCKET"
echo ""

# Export for Python scripts
export HISTORY_BUCKET
export EMBEDDING_MODEL
export PROJECT_ID

# 0. Ensure Local Dependencies (lightweight only)
echo "--- Step 0: Checking Local Dependencies ---"
pip install -q google-cloud-storage 2>/dev/null || true

# 1. Ensure GCS Bucket Exists with proper permissions
echo ""
echo "--- Step 1: Ensuring GCS Bucket ---"
if gsutil ls -b "gs://${HISTORY_BUCKET}" >/dev/null 2>&1; then
    echo "✓ Bucket gs://${HISTORY_BUCKET} exists"
else
    echo "Creating bucket gs://${HISTORY_BUCKET}..."
    gsutil mb -l "${REGION}" "gs://${HISTORY_BUCKET}"
    echo "✓ Bucket created"
fi

# Ensure current user has full access (fixes 403 permission errors)
CURRENT_USER=$(gcloud config get-value account 2>/dev/null)
if [ -n "$CURRENT_USER" ]; then
    echo "Ensuring bucket permissions for $CURRENT_USER..."
    gsutil iam ch "user:${CURRENT_USER}:objectAdmin" "gs://${HISTORY_BUCKET}" 2>/dev/null || true
    echo "✓ Bucket permissions configured"
fi

# 2. Generate Raw Data Locally (fast, no ML)
echo ""
echo "--- Step 2: Generating Raw Data ---"
echo "NOTE: Only generating raw JSON locally (fast)."
echo "      Embedding computation will happen in Cloud Build."
echo ""

# Check if raw data already exists in GCS
RAW_EXISTS=$(gsutil ls "gs://${HISTORY_BUCKET}/raw/${BENCHMARK_SCALE}/dataset.json" 2>/dev/null && echo "yes" || echo "no")

if [ "$RAW_EXISTS" = "yes" ]; then
    echo "✓ Raw data already exists in GCS"
else
    echo "Generating raw synthetic data locally..."
    python3 -m benchmark.generate_and_upload \
        --scale "$BENCHMARK_SCALE" \
        --skip-embeddings \
        --force
    echo "✓ Raw data uploaded to GCS"
fi

# 3. Infrastructure Setup
echo ""
echo "--- Step 3: Infrastructure Setup ---"
./infrastructure.sh

# 4. Build Container & Compute Embeddings in Cloud Build
echo ""
echo "--- Step 4: Cloud Build (Container + Embeddings) ---"
echo "Building container AND computing embeddings in Cloud Build..."
echo "This runs on Google's cloud infrastructure, not your local machine."
echo ""

IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"

# Create a cloudbuild.yaml that builds the container AND computes embeddings
cat > /tmp/cloudbuild-with-embeddings.yaml << EOF
steps:
  # Step 1: Build the container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${IMAGE_URI}', '.']

  # Step 2: Push the container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${IMAGE_URI}']

  # Step 3: Compute embeddings using the built container (runs in cloud!)
  - name: '${IMAGE_URI}'
    entrypoint: 'python3'
    args:
      - '-m'
      - 'benchmark.generate_and_upload'
      - '--ensure-embeddings'
      - '--scale'
      - '${BENCHMARK_SCALE}'
      - '--model'
      - '${EMBEDDING_MODEL}'
    env:
      - 'HISTORY_BUCKET=${HISTORY_BUCKET}'

images:
  - '${IMAGE_URI}'

timeout: '3600s'
options:
  machineType: 'E2_HIGHCPU_8'
EOF

echo "Submitting to Cloud Build (this may take several minutes)..."
gcloud builds submit --config=/tmp/cloudbuild-with-embeddings.yaml .

echo "✓ Container built and embeddings computed in cloud"

# 5. Deploy Services
echo ""
echo "--- Step 5: Deploying Services ---"
chmod +x deploy_commands.sh
./deploy_commands.sh

# 6. Get Controller URL
echo ""
echo "--- Step 6: Resolving Endpoint ---"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)')
echo "Controller URL: $CONTROLLER_URL"

# 7. Run Benchmark
echo ""
echo "--- Step 7: Running Benchmark ---"
echo "Targeting: ${CONTROLLER_URL}"
echo "Scale: ${BENCHMARK_SCALE}"
echo ""

# Export for benchmark script
export CONTROLLER_URL="${CONTROLLER_URL}"
export HISTORY_SCALE="${BENCHMARK_SCALE}"

python3 -m benchmark.research_benchmark

# 8. Report
echo ""
echo "--- Step 8: Results ---"
echo "Results available in $RESULTS_DIR/"

# 9. Cleanup - Delete Cloud Run services to avoid charges
echo ""
echo "--- Step 9: Cleanup (Deleting Cloud Run Services) ---"
echo "Deleting Cloud Run services to avoid ongoing charges..."
echo "(GCS data is preserved for future runs)"
echo ""

# Delete services (ignore errors if they don't exist)
gcloud run services delete dpr-active-controller --region=$REGION --quiet 2>/dev/null || true
gcloud run services delete dpr-passive-workers --region=$REGION --quiet 2>/dev/null || true
gcloud run services delete dpr-baseline-rag --region=$REGION --quiet 2>/dev/null || true

# Also delete the Memorystore Redis instance if it exists (expensive!)
REDIS_INSTANCE="dpr-redis"
if gcloud redis instances describe $REDIS_INSTANCE --region=$REGION >/dev/null 2>&1; then
    echo "Deleting Redis instance (this takes a few minutes)..."
    gcloud redis instances delete $REDIS_INSTANCE --region=$REGION --quiet --async
    echo "Redis deletion initiated (async)"
fi

echo ""
echo "✓ Cloud Run services deleted"
echo "✓ GCS bucket gs://${HISTORY_BUCKET}/ preserved (raw data + embeddings)"
echo ""
echo "=== DPR-RC Cloud Benchmark Complete ==="
echo ""
echo "To re-run benchmarks, simply run ./run_cloud_benchmark.sh again."
echo "The script will reuse existing GCS data and redeploy services."
