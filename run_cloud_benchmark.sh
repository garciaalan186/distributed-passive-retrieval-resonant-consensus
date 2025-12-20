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
DEBUG_BREAKPOINTS="${DEBUG_BREAKPOINTS:-false}"
DEBUG_PAUSE_SECONDS="${DEBUG_PAUSE_SECONDS:-2}"

echo "=== DPR-RC Cloud Benchmark Start ==="
echo "NOTE: All Python/ML work runs in Google Cloud, not locally."
echo ""

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
echo "Project: $PROJECT_ID"
echo "Scale: $BENCHMARK_SCALE"
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Region: $REGION"
echo "Debug Mode: $DEBUG_BREAKPOINTS (pause: ${DEBUG_PAUSE_SECONDS}s)"
echo ""

if [ "$DEBUG_BREAKPOINTS" = "true" ]; then
    echo "🔍 DEBUG MODE ENABLED"
    echo "   Each step in the pipeline will log detailed output."
    echo "   Use 'gcloud logs tail' to watch the debug output in real-time."
    echo ""
fi

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
# Grant Compute Engine default service account access (used by Cloud Build workers)
gsutil iam ch "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com:objectAdmin" "gs://${HISTORY_BUCKET}" 2>/dev/null || true

# CRITICAL FIX: Grant Cloud Run service account access to GCS bucket
# Cloud Run services use the default Compute Engine service account
echo "Granting Cloud Run service account GCS access..."
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
# Grant objectAdmin for full read/write access
gsutil iam ch "serviceAccount:${COMPUTE_SA}:objectAdmin" "gs://${HISTORY_BUCKET}" 2>/dev/null || true
# Also try roles/storage.objectAdmin for compatibility
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.objectAdmin" \
  --condition=None 2>/dev/null || true

echo "✓ Bucket permissions configured (Cloud Build + Cloud Run)"

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
      # Removed --force to allow caching
    env:
      - 'HISTORY_BUCKET=${HISTORY_BUCKET}'
      - 'TOKENIZERS_PARALLELISM=false'
    dir: '/app'
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
    echo "No deploy_commands.sh found. Deploying manually (HTTP mode - no Redis)..."

    # Step 4a: Deploy SLM Service first (other services depend on it)
    echo "Deploying SLM Service..."
    gcloud run deploy dpr-slm-service \
        --image="${IMAGE_URI}" \
        --region="${REGION}" \
        --set-env-vars="ROLE=slm,SLM_MODEL=Qwen/Qwen2-0.5B-Instruct" \
        --memory=4Gi \
        --cpu=2 \
        --timeout=300 \
        --min-instances=1 \
        --allow-unauthenticated \
        --quiet || true

    # Get SLM service URL
    SLM_SERVICE_URL=$(gcloud run services describe dpr-slm-service --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
    echo "SLM Service URL: $SLM_SERVICE_URL"

    # Step 4b: Deploy Passive Workers FIRST (Active Controller needs their URL)
    echo "Deploying Passive Workers..."
    gcloud run deploy dpr-passive-worker \
        --image="${IMAGE_URI}" \
        --region="${REGION}" \
        --set-env-vars="ROLE=passive,HISTORY_BUCKET=${HISTORY_BUCKET},HISTORY_SCALE=${BENCHMARK_SCALE},SLM_SERVICE_URL=${SLM_SERVICE_URL},DEBUG_BREAKPOINTS=${DEBUG_BREAKPOINTS},DEBUG_PAUSE_SECONDS=${DEBUG_PAUSE_SECONDS}" \
        --memory=2Gi \
        --min-instances=1 \
        --timeout=300 \
        --allow-unauthenticated \
        --quiet || true

    # Get Passive Worker URL
    PASSIVE_WORKER_URL=$(gcloud run services describe dpr-passive-worker --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
    echo "Passive Worker URL: $PASSIVE_WORKER_URL"

    # Step 4c: Deploy Active Controller (with HTTP worker URL instead of Redis)
    echo "Deploying Active Controller..."
    gcloud run deploy dpr-active-controller \
        --image="${IMAGE_URI}" \
        --region="${REGION}" \
        --allow-unauthenticated \
        --set-env-vars="ROLE=active,HISTORY_BUCKET=${HISTORY_BUCKET},HISTORY_SCALE=${BENCHMARK_SCALE},SLM_SERVICE_URL=${SLM_SERVICE_URL},ENABLE_QUERY_ENHANCEMENT=true,PASSIVE_WORKER_URL=${PASSIVE_WORKER_URL},USE_HTTP_WORKERS=true,DEBUG_BREAKPOINTS=${DEBUG_BREAKPOINTS},DEBUG_PAUSE_SECONDS=${DEBUG_PAUSE_SECONDS}" \
        --memory=2Gi \
        --timeout=300 \
        --quiet || true
fi

# --- Step 5: Get Service URLs ---
echo ""
echo "--- Step 5: Resolving Endpoints ---"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
SLM_SERVICE_URL=$(gcloud run services describe dpr-slm-service --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
WORKER_URL=$(gcloud run services describe dpr-passive-worker --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")

if [ -z "$CONTROLLER_URL" ]; then
    echo "Warning: Could not get controller URL. Services may not be deployed."
    echo "Skipping benchmark execution."
else
    echo "Controller URL: $CONTROLLER_URL"
    echo "SLM Service URL: $SLM_SERVICE_URL"
    echo "Worker URL: $WORKER_URL"

    # --- Step 5.5: Verify All Services Are Ready ---
    echo ""
    echo "--- Step 5.5: Verifying Service Readiness ---"

    # Check SLM service is ready
    echo "Checking SLM service..."
    MAX_WAIT=300
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SLM_SERVICE_URL}/ready" 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            echo "✓ SLM service is ready"
            break
        fi
        if [ $ELAPSED -eq 0 ]; then
            echo "  Waiting for SLM service to load model (this may take 3-5 minutes)..."
        fi
        sleep 10
        ELAPSED=$((ELAPSED + 10))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "✗ ERROR: SLM service did not become ready within ${MAX_WAIT}s"
        echo "  Cannot run benchmark with SLM service unavailable"
        exit 1
    fi

    # Check Controller service is ready
    echo "Checking Active Controller..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${CONTROLLER_URL}/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✓ Active Controller is ready"
    else
        echo "⚠ WARNING: Active Controller health check returned HTTP $HTTP_CODE"
    fi

    # Check Worker service is ready
    echo "Checking Passive Worker..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${WORKER_URL}/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✓ Passive Worker is ready"
    else
        echo "⚠ WARNING: Passive Worker health check returned HTTP $HTTP_CODE"
    fi

    echo ""
    echo "All critical services verified. Starting benchmark..."

    # --- Step 6: Run Benchmark in Cloud Build ---
    echo ""
    echo "--- Step 6: Running Benchmark (in Cloud Build) ---"
    echo "Executing benchmark from cloud against deployed services..."

    if [ "$DEBUG_BREAKPOINTS" = "true" ]; then
        echo ""
        echo "📋 To watch debug logs in real-time, open another terminal and run:"
        echo "   gcloud logging read 'resource.type=cloud_run_revision AND (resource.labels.service_name=dpr-active-controller OR resource.labels.service_name=dpr-passive-worker)' --project=$PROJECT_ID --format='value(textPayload)' --freshness=5m | grep -A50 'DEBUG BREAKPOINT'"
        echo ""
        echo "Or stream logs continuously:"
        echo "   gcloud alpha logging tail 'resource.type=cloud_run_revision' --project=$PROJECT_ID --format='value(textPayload)' 2>/dev/null | grep --line-buffered -E 'DEBUG|BREAKPOINT|Request|Response|Vote|Consensus'"
        echo ""
    fi

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
      - 'SLM_SERVICE_URL=${SLM_SERVICE_URL}'
      - 'HISTORY_BUCKET=${HISTORY_BUCKET}'
      - 'HISTORY_SCALE=${BENCHMARK_SCALE}'
      - 'BENCHMARK_SCALE=${BENCHMARK_SCALE}'
      - 'TOKENIZERS_PARALLELISM=false'
      - 'ANONYMIZED_TELEMETRY=false'
    dir: '/app'
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

# --- Step 7.5: Download Debug Logs (before service deletion) ---
echo ""
echo "--- Step 7.5: Downloading Debug Logs ---"
LOG_FILE="${RESULTS_DIR}/debug_logs_${BENCHMARK_SCALE}_$(date +%Y%m%d_%H%M%S).txt"

if [ "$DEBUG_BREAKPOINTS" = "true" ]; then
    echo "Downloading debug logs from Cloud Logging..."
    gcloud logging read \
        "resource.type=cloud_run_revision AND (resource.labels.service_name=dpr-active-controller OR resource.labels.service_name=dpr-passive-worker OR resource.labels.service_name=dpr-slm-service) AND timestamp>=\"$(date -u -d '30 minutes ago' '+%Y-%m-%dT%H:%M:%SZ')\"" \
        --project="$PROJECT_ID" \
        --format='value(timestamp, resource.labels.service_name, severity, textPayload)' \
        --limit=10000 \
        > "$LOG_FILE" 2>/dev/null || true

    if [ -s "$LOG_FILE" ]; then
        echo "✓ Debug logs saved to: $LOG_FILE"
        LOG_SIZE=$(wc -l < "$LOG_FILE" | tr -d ' ')
        echo "  ($LOG_SIZE log entries)"
    else
        echo "⚠ No debug logs found (services may not have generated debug output)"
        rm -f "$LOG_FILE"
    fi
else
    echo "Downloading service logs (last 30 minutes)..."
    gcloud logging read \
        "resource.type=cloud_run_revision AND (resource.labels.service_name=dpr-active-controller OR resource.labels.service_name=dpr-passive-worker OR resource.labels.service_name=dpr-slm-service) AND timestamp>=\"$(date -u -d '30 minutes ago' '+%Y-%m-%dT%H:%M:%SZ')\"" \
        --project="$PROJECT_ID" \
        --format='value(timestamp, resource.labels.service_name, severity, textPayload)' \
        --limit=5000 \
        > "$LOG_FILE" 2>/dev/null || true

    if [ -s "$LOG_FILE" ]; then
        echo "✓ Service logs saved to: $LOG_FILE"
        LOG_SIZE=$(wc -l < "$LOG_FILE" | tr -d ' ')
        echo "  ($LOG_SIZE log entries)"
    else
        echo "⚠ No service logs found"
        rm -f "$LOG_FILE"
    fi
fi

# --- Step 8: Cleanup ---
echo ""
echo "--- Step 8: Cleanup Options ---"
echo ""
echo "Services are currently running:"
echo "  - SLM Service (GPU): ~\$18/day"
echo "  - Active Controller: ~\$1/day"
echo "  - Passive Worker: ~\$1/day"
echo ""
echo "Options:"
echo "  [1] Keep services running for rapid iteration (auto-teardown in 15 min)"
echo "  [2] Tear down services now"
echo ""

# Read user input with 15-minute timeout (900 seconds)
read -t 900 -p "Choose option [1 or 2] (default: teardown in 15min): " CLEANUP_CHOICE

# Default to teardown if timeout or invalid input
if [ -z "$CLEANUP_CHOICE" ]; then
    echo ""
    echo "⏱  15-minute timer expired. Initiating teardown..."
    CLEANUP_CHOICE="2"
elif [ "$CLEANUP_CHOICE" = "1" ]; then
    echo ""
    echo "✓ Keeping services running for rapid iteration"
    echo "  Services will remain active for testing"
    echo "  Run this script again to execute another benchmark"
    echo ""
    echo "⚠  REMINDER: Run the following when done to avoid charges:"
    echo "    gcloud run services delete dpr-active-controller --region=$REGION --quiet"
    echo "    gcloud run services delete dpr-passive-worker --region=$REGION --quiet"
    echo "    gcloud run services delete dpr-slm-service --region=$REGION --quiet"
    echo ""
    # Skip cleanup entirely
    CLEANUP_CHOICE="skip"
fi

if [ "$CLEANUP_CHOICE" != "skip" ]; then
    echo ""
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
fi

echo "✓ GCS bucket gs://${HISTORY_BUCKET}/ preserved (raw data + embeddings)"
if [ -f "$LOG_FILE" ]; then
    echo "✓ Debug logs saved to: $LOG_FILE"
fi
echo ""
echo "=== DPR-RC Cloud Benchmark Complete ==="
echo ""
echo "Results Location:"
echo "  - Benchmark outputs: ${RESULTS_DIR}/"
if [ -f "$LOG_FILE" ]; then
    echo "  - Service logs: $LOG_FILE"
fi
echo ""
echo "To re-run benchmarks, simply run ./run_cloud_benchmark.sh again."
echo "The script will reuse existing GCS data and redeploy services."
echo ""
echo "Debug Mode Usage:"
echo "  DEBUG_BREAKPOINTS=true ./run_cloud_benchmark.sh"
echo ""
echo "  This enables step-by-step logging at each pipeline edge:"
echo "    1. Query Received (Client → Controller)"
echo "    2. Query Enhancement (Controller → SLM)"
echo "    3. L1 Routing (Controller → Router)"
echo "    4. HTTP Worker Call (Controller → Worker)"
echo "    5. Shard Loading (Worker → GCS)"
echo "    6. Document Retrieval (Worker → ChromaDB)"
echo "    7. L2 Verification (Worker → SLM)"
echo "    8. L3 Quadrant Calculation"
echo "    9. Vote Creation (Worker → Controller)"
echo "   10. Consensus Calculation"
echo "   11. Final Response (Controller → Client)"
echo ""
echo "  View debug logs with:"
echo "    ./view_debug_logs.sh"
