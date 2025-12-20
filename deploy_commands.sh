#!/bin/bash
# deploy_commands.sh
# HTTP-based deployment for Cloud Run (no Redis dependency)

set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="us-central1"
IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"
HISTORY_BUCKET="dpr-history-data-${PROJECT_ID}"
BENCHMARK_SCALE="${BENCHMARK_SCALE:-medium}"
DEBUG_BREAKPOINTS="${DEBUG_BREAKPOINTS:-false}"
DEBUG_PAUSE_SECONDS="${DEBUG_PAUSE_SECONDS:-2}"

echo "Deploying DPR-RC services in HTTP mode (no Redis)"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $HISTORY_BUCKET"
echo "Scale: $BENCHMARK_SCALE"
echo ""

# Step 1: Deploy SLM Service first (other services depend on it)
echo "Deploying SLM Service with GPU acceleration..."
gcloud run deploy dpr-slm-service \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --set-env-vars="ROLE=slm,SLM_MODEL=Qwen/Qwen2-0.5B-Instruct" \
    --memory=16Gi \
    --cpu=4 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --no-cpu-throttling \
    --no-gpu-zonal-redundancy \
    --timeout=300 \
    --min-instances=1 \
    --allow-unauthenticated \
    --quiet

# Get SLM service URL
SLM_SERVICE_URL=$(gcloud run services describe dpr-slm-service --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
echo "SLM Service URL: $SLM_SERVICE_URL"

# Wait for SLM service to be ready (GPU model loading ~30-60s including warmup)
echo "Waiting for SLM service to be ready (GPU model loading + warmup)..."
MAX_WAIT=300  # 5 minutes max
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SLM_SERVICE_URL}/ready" || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✓ SLM service is ready"
        break
    fi
    echo "  SLM service not ready yet (HTTP $HTTP_CODE), waiting... (${ELAPSED}s/${MAX_WAIT}s)"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "⚠ WARNING: SLM service did not become ready within ${MAX_WAIT}s"
    echo "  Continuing anyway, but queries may fail until model finishes loading"
fi

# Step 2: Deploy Passive Workers FIRST (Active Controller needs their URL)
echo "Deploying Passive Workers..."
gcloud run deploy dpr-passive-worker \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --set-env-vars="ROLE=passive,HISTORY_BUCKET=${HISTORY_BUCKET},HISTORY_SCALE=${BENCHMARK_SCALE},SLM_SERVICE_URL=${SLM_SERVICE_URL},DEBUG_BREAKPOINTS=${DEBUG_BREAKPOINTS},DEBUG_PAUSE_SECONDS=${DEBUG_PAUSE_SECONDS}" \
    --memory=2Gi \
    --min-instances=1 \
    --timeout=300 \
    --allow-unauthenticated \
    --quiet

# Get Passive Worker URL
PASSIVE_WORKER_URL=$(gcloud run services describe dpr-passive-worker --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
echo "Passive Worker URL: $PASSIVE_WORKER_URL"

# Step 3: Deploy Active Controller (with HTTP worker URL instead of Redis)
echo "Deploying Active Controller..."
gcloud run deploy dpr-active-controller \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --allow-unauthenticated \
    --set-env-vars="ROLE=active,HISTORY_BUCKET=${HISTORY_BUCKET},HISTORY_SCALE=${BENCHMARK_SCALE},SLM_SERVICE_URL=${SLM_SERVICE_URL},ENABLE_QUERY_ENHANCEMENT=true,PASSIVE_WORKER_URL=${PASSIVE_WORKER_URL},USE_HTTP_WORKERS=true,DEBUG_BREAKPOINTS=${DEBUG_BREAKPOINTS},DEBUG_PAUSE_SECONDS=${DEBUG_PAUSE_SECONDS}" \
    --memory=2Gi \
    --min-instances=1 \
    --timeout=300 \
    --quiet

echo ""
echo "✓ Deployment complete"
echo ""
echo "Service URLs:"
echo "  SLM: $SLM_SERVICE_URL"
echo "  Worker: $PASSIVE_WORKER_URL"
echo "  Controller: $(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)' 2>/dev/null)"
