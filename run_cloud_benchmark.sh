#!/bin/bash
# run_cloud_benchmark.sh
# End-to-end cloud benchmark - NO Python runs locally
#
# ALL work happens in the cloud:
# - Data generation: Cloud Build
# - Embedding computation: Cloud Build
# - Benchmark execution: Cloud Build
#
# Your local machine only runs gcloud/gsutil commands.

set -e

RESULTS_DIR="benchmark_results_cloud"
mkdir -p $RESULTS_DIR

# Configuration
BENCHMARK_SCALE="${BENCHMARK_SCALE:-medium}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
REGION="${REGION:-us-central1}"

echo "=== DPR-RC Cloud Benchmark Start ==="
echo "NOTE: All Python/ML work runs in Google Cloud, not locally."
echo ""

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
echo "Project: $PROJECT_ID"
echo "Scale: $BENCHMARK_SCALE"
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Region: $REGION"
echo ""

# Set bucket name based on project ID for uniqueness
HISTORY_BUCKET="dpr-history-data-${PROJECT_ID}"
echo "History Bucket: $HISTORY_BUCKET"
echo ""

IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"

# --- Step 1: Ensure GCS Bucket Exists ---
echo "--- Step 1: Ensuring GCS Bucket ---"
if gsutil ls -b "gs://${HISTORY_BUCKET}" >/dev/null 2>&1; then
    echo "✓ Bucket gs://${HISTORY_BUCKET} exists"
else
    echo "Creating bucket gs://${HISTORY_BUCKET}..."
    gsutil mb -l "${REGION}" "gs://${HISTORY_BUCKET}"
    echo "✓ Bucket created"
fi

# Ensure current user and Cloud Build have access
CURRENT_USER=$(gcloud config get-value account 2>/dev/null)
if [ -n "$CURRENT_USER" ]; then
    echo "Configuring bucket permissions..."
    gsutil iam ch "user:${CURRENT_USER}:objectAdmin" "gs://${HISTORY_BUCKET}" 2>/dev/null || true
fi
# Grant Cloud Build service account access
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gsutil iam ch "serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com:objectAdmin" "gs://${HISTORY_BUCKET}" 2>/dev/null || true
echo "✓ Bucket permissions configured"

# --- Step 2: Infrastructure Setup ---
echo ""
echo "--- Step 2: Infrastructure Setup ---"
if [ -f "./infrastructure.sh" ]; then
    chmod +x infrastructure.sh
    ./infrastructure.sh
else
    echo "No infrastructure.sh found, skipping..."
fi

# --- Step 3: Cloud Build - Build, Generate Data, Compute Embeddings ---
echo ""
echo "--- Step 3: Cloud Build (Container + Data + Embeddings) ---"
echo "Building container and preparing all data in Google Cloud..."
echo "This runs on cloud infrastructure, not your local machine."
echo ""

# Create cloudbuild.yaml for full data prep
cat > /tmp/cloudbuild-full.yaml << CLOUDEOF
steps:
  # Step 1: Build the container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${IMAGE_URI}', '.']
    id: 'build'

  # Step 2: Push the container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${IMAGE_URI}']
    id: 'push'
    waitFor: ['build']

  # Step 3: Generate raw data AND compute embeddings (all in cloud)
  - name: '${IMAGE_URI}'
    entrypoint: 'python3'
    args:
      - '-m'
      - 'benchmark.generate_and_upload'
      - '--scale'
      - '${BENCHMARK_SCALE}'
      - '--model'
      - '${EMBEDDING_MODEL}'
      - '--force'
    env:
      - 'HISTORY_BUCKET=${HISTORY_BUCKET}'
      - 'TOKENIZERS_PARALLELISM=false'
    id: 'generate-data'
    waitFor: ['push']

images:
  - '${IMAGE_URI}'

timeout: '3600s'
options:
  machineType: 'E2_HIGHCPU_8'
CLOUDEOF

echo "Submitting data preparation to Cloud Build..."
gcloud builds submit --config=/tmp/cloudbuild-full.yaml . --timeout=3600s

echo "✓ Container built, data generated, and embeddings computed in cloud"

# --- Step 4: Deploy Services ---
echo ""
echo "--- Step 4: Deploying Services ---"
if [ -f "./deploy_commands.sh" ]; then
    chmod +x deploy_commands.sh
    ./deploy_commands.sh
else
    echo "No deploy_commands.sh found. Deploying manually..."

    # Step 4a: Deploy SLM Service first (passive workers depend on it)
    echo "Deploying SLM Service..."
    gcloud run deploy dpr-slm-service \
        --image="${IMAGE_URI}" \
        --region="${REGION}" \
        --set-env-vars="ROLE=slm,SLM_MODEL=Qwen/Qwen2-0.5B-Instruct" \
        --memory=4Gi \
        --cpu=2 \
        --timeout=300 \
        --min-instances=1 \
        --no-allow-unauthenticated \
        --quiet || true

    # Get SLM service URL
    SLM_SERVICE_URL=$(gcloud run services describe dpr-slm-service --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
    echo "SLM Service URL: $SLM_SERVICE_URL"

    # Step 4b: Deploy Active Controller (with SLM for query enhancement)
    echo "Deploying Active Controller..."
    gcloud run deploy dpr-active-controller \
        --image="${IMAGE_URI}" \
        --region="${REGION}" \
        --allow-unauthenticated \
        --set-env-vars="ROLE=active,HISTORY_BUCKET=${HISTORY_BUCKET},HISTORY_SCALE=${BENCHMARK_SCALE},SLM_SERVICE_URL=${SLM_SERVICE_URL},ENABLE_QUERY_ENHANCEMENT=true" \
        --memory=2Gi \
        --timeout=300 \
        --quiet || true

    # Step 4c: Deploy Passive Workers (with SLM_SERVICE_URL)
    if [ -n "$SLM_SERVICE_URL" ]; then
        echo "Deploying Passive Workers..."
        gcloud run deploy dpr-passive-worker \
            --image="${IMAGE_URI}" \
            --region="${REGION}" \
            --set-env-vars="ROLE=passive,HISTORY_BUCKET=${HISTORY_BUCKET},HISTORY_SCALE=${BENCHMARK_SCALE},SLM_SERVICE_URL=${SLM_SERVICE_URL}" \
            --memory=2Gi \
            --min-instances=3 \
            --no-allow-unauthenticated \
            --quiet || true
    fi
fi

# --- Step 5: Get Controller URL ---
echo ""
echo "--- Step 5: Resolving Endpoint ---"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")

if [ -z "$CONTROLLER_URL" ]; then
    echo "Warning: Could not get controller URL. Services may not be deployed."
    echo "Skipping benchmark execution."
else
    echo "Controller URL: $CONTROLLER_URL"

    # --- Step 6: Run Benchmark in Cloud Build ---
    echo ""
    echo "--- Step 6: Running Benchmark (in Cloud Build) ---"
    echo "Executing benchmark from cloud against deployed services..."

    # Create cloudbuild for benchmark execution
    cat > /tmp/cloudbuild-benchmark.yaml << BENCHEOF
steps:
  - name: '${IMAGE_URI}'
    entrypoint: 'python3'
    args:
      - '-m'
      - 'benchmark.research_benchmark'
    env:
      - 'CONTROLLER_URL=${CONTROLLER_URL}'
      - 'HISTORY_BUCKET=${HISTORY_BUCKET}'
      - 'HISTORY_SCALE=${BENCHMARK_SCALE}'
      - 'TOKENIZERS_PARALLELISM=false'
    id: 'run-benchmark'

timeout: '1800s'
options:
  machineType: 'E2_HIGHCPU_8'
BENCHEOF

    echo "Submitting benchmark to Cloud Build..."
    gcloud builds submit --config=/tmp/cloudbuild-benchmark.yaml --no-source --timeout=1800s

    echo "✓ Benchmark completed in cloud"
fi

# --- Step 7: Download Results ---
echo ""
echo "--- Step 7: Downloading Results ---"
# Results are saved locally by the benchmark, but since it ran in cloud,
# we need to check if there are any results to download from GCS
if gsutil ls "gs://${HISTORY_BUCKET}/benchmark_results/" >/dev/null 2>&1; then
    echo "Downloading benchmark results from GCS..."
    gsutil -m cp -r "gs://${HISTORY_BUCKET}/benchmark_results/*" "${RESULTS_DIR}/" 2>/dev/null || true
    echo "✓ Results downloaded to ${RESULTS_DIR}/"
else
    echo "Note: Results are in Cloud Build logs (benchmark ran in cloud)"
fi

# --- Step 8: Cleanup ---
echo ""
echo "--- Step 8: Cleanup (Deleting Cloud Run Services) ---"
echo "Deleting Cloud Run services to avoid ongoing charges..."
echo "(GCS data is preserved for future runs)"
echo ""

# Delete services (ignore errors if they don't exist)
gcloud run services delete dpr-active-controller --region=$REGION --quiet 2>/dev/null || true
gcloud run services delete dpr-passive-worker --region=$REGION --quiet 2>/dev/null || true
gcloud run services delete dpr-passive-workers --region=$REGION --quiet 2>/dev/null || true
gcloud run services delete dpr-slm-service --region=$REGION --quiet 2>/dev/null || true
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
