#!/bin/bash
# infrastructure.sh
# Provisions Google Cloud Platform resources for DPR-RC Architecture

set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REDIS_NAME="dpr-redis"
REPO_NAME="dpr-repo"
BUCKET_NAME="dpr-audit-logs-${PROJECT_ID}"
HISTORY_BUCKET_NAME="dpr-history-data-${PROJECT_ID}"
SERVICE_ACCOUNT="dpr-agent-sa"

echo "Provisioning Infrastructure for Project: $PROJECT_ID in $REGION"

# 1. Enable APIs
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    redis.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    logging.googleapis.com \
    cloudbuild.googleapis.com \
    compute.googleapis.com \
    vpcaccess.googleapis.com

# 2. Storage Setup
echo "Creating GCS Buckets..."
gsutil mb -l $REGION gs://${BUCKET_NAME} || echo "Bucket ${BUCKET_NAME} may already exist."
gsutil uniformbucketlevelaccess set on gs://${BUCKET_NAME} || echo "Setting UBLA failed or already set."

gsutil mb -l $REGION gs://${HISTORY_BUCKET_NAME} || echo "Bucket ${HISTORY_BUCKET_NAME} may already exist."
gsutil uniformbucketlevelaccess set on gs://${HISTORY_BUCKET_NAME} || echo "Setting UBLA failed or already set."

# 3. Redis Setup (Messages & State)
echo "Creating Cloud Memorystore for Redis..."
gcloud redis instances create $REDIS_NAME \
    --size=1 \
    --region=$REGION \
    --redis-version=redis_6_x \
    --tier=basic \
    --network=default \
    || echo "Redis instance $REDIS_NAME may already exist."

# Wait for Redis to be READY
echo "Waiting for Redis to be READY..."
while true; do
    STATE=$(gcloud redis instances describe $REDIS_NAME --region=$REGION --format="value(state)" 2>/dev/null || echo "UNKNOWN")
    if [ "$STATE" = "READY" ]; then
        echo "Redis is READY."
        break
    fi
    echo "Redis state: $STATE. Waiting 10s..."
    sleep 10
done

# Capture Redis Host/Port
REDIS_HOST=$(gcloud redis instances describe $REDIS_NAME --region=$REGION --format="value(host)")
REDIS_PORT=$(gcloud redis instances describe $REDIS_NAME --region=$REGION --format="value(port)")
echo "Redis instance created at $REDIS_HOST:$REDIS_PORT"

# 3a. VPC Connector for Cloud Run to access Redis
echo "Creating VPC Connector for Cloud Run..."
gcloud compute networks vpc-access connectors create dpr-vpc-connector \
    --region=$REGION \
    --network=default \
    --range=10.8.0.0/28 \
    || echo "VPC Connector may already exist, continuing..."

# 4. Artifact Registry
echo "Creating Artifact Registry..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="DPR-RC Container Repository" \
    || echo "Repository $REPO_NAME may already exist."

# 5. IAM Setup
echo "Creating Service Account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT \
    --display-name="DPR RC Agent Service Account" \
    || echo "Service Account $SERVICE_ACCOUNT may already exist."

SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.invoker"

# 6. Build & Deploy Placeholders (Requires Dockerfile present)
# Note: Actual deployment happens after code generation.
# This section generates the command for the user to run later.

cat <<EOF > deploy_commands.sh
#!/bin/bash
# Deploy Active Controller
gcloud run deploy dpr-active-controller \\
    --image=gcr.io/${PROJECT_ID}/dpr-agent:latest \\
    --region=${REGION} \\
    --service-account=${SA_EMAIL} \\
    --set-env-vars="REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},LOG_BUCKET=${BUCKET_NAME},ROLE=active,CONTROLLER_URL=http://localhost:8080/query" \\
    --vpc-connector=dpr-vpc-connector \\
    --vpc-egress=private-ranges-only \\
    --allow-unauthenticated

# Deploy Passive Worker (Scale to 3 minimum for consensus)
gcloud run deploy dpr-passive-worker \\
    --image=gcr.io/${PROJECT_ID}/dpr-agent:latest \\
    --region=${REGION} \\
    --service-account=${SA_EMAIL} \\
    --set-env-vars="REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},LOG_BUCKET=${BUCKET_NAME},ROLE=passive,HISTORY_BUCKET=${HISTORY_BUCKET_NAME},DATA_SCALE=medium" \\
    --vpc-connector=dpr-vpc-connector \\
    --vpc-egress=private-ranges-only \\
    --min-instances=3 \\
    --no-allow-unauthenticated

# Deploy Baseline RAG (for fair cloud-to-cloud comparison)
gcloud run deploy dpr-baseline-rag \\
    --image=gcr.io/${PROJECT_ID}/dpr-agent:latest \\
    --region=${REGION} \\
    --service-account=${SA_EMAIL} \\
    --set-env-vars="ROLE=baseline,HISTORY_BUCKET=${HISTORY_BUCKET_NAME},DATA_SCALE=medium" \\
    --allow-unauthenticated
EOF

chmod +x deploy_commands.sh
echo "Infrastructure provisioning complete. Run './deploy_commands.sh' after building the container."
