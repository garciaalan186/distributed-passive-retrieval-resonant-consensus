#!/bin/bash
# DPR-RC Service Deployment
# Order matters: SLM service must be deployed before passive workers

set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION=${REGION:-us-central1}
IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"
SERVICE_ACCOUNT="dpr-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"
VPC_CONNECTOR="dpr-vpc-connector"
REDIS_HOST=${REDIS_HOST:-"10.6.246.99"}
REDIS_PORT=${REDIS_PORT:-6379}
LOG_BUCKET="dpr-audit-logs-${PROJECT_ID}"
HISTORY_BUCKET="dpr-history-data-${PROJECT_ID}"

echo "=== Deploying DPR-RC Services ==="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"

# --- Step 1: Deploy SLM Service (must be first - other services depend on it) ---
echo ""
echo "--- Deploying SLM Service ---"
gcloud run deploy dpr-slm-service \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="ROLE=slm,SLM_MODEL=Qwen/Qwen2-0.5B-Instruct,SLM_PORT=8080" \
    --vpc-connector="${VPC_CONNECTOR}" \
    --vpc-egress=private-ranges-only \
    --memory=4Gi \
    --cpu=2 \
    --timeout=300 \
    --min-instances=1 \
    --max-instances=3 \
    --no-allow-unauthenticated

# Get SLM service URL for passive workers
SLM_SERVICE_URL=$(gcloud run services describe dpr-slm-service --region="${REGION}" --format='value(status.url)')
echo "SLM Service URL: ${SLM_SERVICE_URL}"

# --- Step 2: Deploy Active Controller ---
echo ""
echo "--- Deploying Active Controller ---"
gcloud run deploy dpr-active-controller \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},LOG_BUCKET=${LOG_BUCKET},ROLE=active,CONTROLLER_URL=http://localhost:8080/query" \
    --vpc-connector="${VPC_CONNECTOR}" \
    --vpc-egress=private-ranges-only \
    --memory=2Gi \
    --timeout=300 \
    --allow-unauthenticated

# --- Step 3: Deploy Passive Worker (Scale to 3 minimum for consensus) ---
echo ""
echo "--- Deploying Passive Workers ---"
gcloud run deploy dpr-passive-worker \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},LOG_BUCKET=${LOG_BUCKET},ROLE=passive,HISTORY_BUCKET=${HISTORY_BUCKET},SLM_SERVICE_URL=${SLM_SERVICE_URL}" \
    --vpc-connector="${VPC_CONNECTOR}" \
    --vpc-egress=private-ranges-only \
    --memory=2Gi \
    --min-instances=3 \
    --no-allow-unauthenticated

echo ""
echo "=== Deployment Complete ==="
echo "SLM Service: ${SLM_SERVICE_URL}"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region="${REGION}" --format='value(status.url)')
echo "Active Controller: ${CONTROLLER_URL}"
