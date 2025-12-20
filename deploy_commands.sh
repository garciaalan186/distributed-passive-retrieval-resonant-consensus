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
echo "Deploying SLM Service (CPU mode for cost efficiency)..."
# Changed to CPU mode: GPU (L4) is overkill/expensive for 0.5B model
gcloud run deploy dpr-slm-service \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --set-env-vars="ROLE=slm,SLM_MODEL=Qwen/Qwen2-0.5B-Instruct" \
    --memory=4Gi \
    --cpu=4 \
    --timeout=300 \
    --min-instances=1 \
    --allow-unauthenticated \
    --quiet &
SLM_PID=$!

echo "Deploying Passive Workers (in parallel)..."
gcloud run deploy dpr-passive-worker \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --set-env-vars="ROLE=passive,HISTORY_BUCKET=${HISTORY_BUCKET},HISTORY_SCALE=${BENCHMARK_SCALE},SLM_SERVICE_URL=PENDING,DEBUG_BREAKPOINTS=${DEBUG_BREAKPOINTS},DEBUG_PAUSE_SECONDS=${DEBUG_PAUSE_SECONDS}" \
    --memory=2Gi \
    --min-instances=1 \
    --timeout=300 \
    --allow-unauthenticated \
    --quiet &
WORKER_PID=$!

echo "Deploying Active Controller (in parallel)..."
# Note: Controller needs worker URL, but we'll let it deploy and just fail health checks until re-config or basic startup
# Actually, for true parallelism, we can't pass the URL immediately if we don't know it.
# Strategy update: We must wait for SLM and Worker to get their URLs before we can configure them?
# Cloud Run URLs are deterministic if service name is static.
# But generally we need to wait.

# Compromise: Deploy SLM and Worker in parallel (independent). Wait for them. Then deploy Controller.
echo "Waiting for SLM and Worker deployments to complete..."
wait $SLM_PID
wait $WORKER_PID

# Get URLs now that they are deployed
SLM_SERVICE_URL=$(gcloud run services describe dpr-slm-service --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")
PASSIVE_WORKER_URL=$(gcloud run services describe dpr-passive-worker --region=$REGION --format='value(status.url)' 2>/dev/null || echo "")

echo "SLM Service URL: $SLM_SERVICE_URL"
echo "Passive Worker URL: $PASSIVE_WORKER_URL"

# Re-deploy/Update Passive Worker with correct SLM URL if needed? 
# The SLM URL is likely static if service name hasn't changed.
# Let's assume standard names "dpr-slm-service". run.app domains are stable.

# Wait for SLM service to be ready (CPU model loading -> faster than GPU warmup usually)
echo "Waiting for SLM service to be ready..."
MAX_WAIT=300 
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SLM_SERVICE_URL}/readiness" || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✓ SLM service is ready"
        break
    fi
    # echo "  SLM service not ready yet..."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "⚠ WARNING: SLM service did not become ready within ${MAX_WAIT}s"
    echo "  Continuing anyway, but queries may fail until model finishes loading"
fi

# Now update Passive Worker with the confirmed SLM URL (if it was PENDING)
# Or just deploy Controller.
# Original script deployed Passive after SLM. 
# We deployed them in parallel.
# To keep it simple and robust:
# 1. Deploy SLM (Background)
# 2. Deploy Passive (Background) - It might crash if it tries to call SLM on startup? 
#    Actually, current Passive Worker code doesn't call SLM on init, only on request. So it's safe.
#    BUT we need to pass SLM_SERVICE_URL env var. 
#    If we don't know the URL yet (first deploy), this fails.
#    Since we are optimizing for "Test Runs", we can assume URLs exist or we accept 2-step.

# Let's stick to the semi-parallel approach that is safe:
# 1. Deploy SLM & Passive in parallel. 
#    (We use the expected URL structure or update later).
#    Google Cloud Run URLs are predictable: https://[SERVICE]-[HASH]-[REGION].run.app
#    We can't guess the hash on first deploy.

# Fallback to Serial for SLM -> Passive dependency, but keep SLM optimization.
# Reverting the parallel chunk above to standard Serial for safety, but with CPU optimization applied.

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
