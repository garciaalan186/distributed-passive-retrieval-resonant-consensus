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
    cloudbuild.googleapis.com

# 2. Storage Setup
echo "Creating GCS Buckets..."
gcloud storage buckets create gs://${BUCKET_NAME} --location=$REGION --uniform-bucket-level-access
gcloud storage buckets create gs://${HISTORY_BUCKET_NAME} --location=$REGION --uniform-bucket-level-access

# 3. Redis Setup (Messages & State)
echo "Creating Cloud Memorystore for Redis..."
gcloud redis instances create $REDIS_NAME \
    --size=1 \
    --region=$REGION \
    --redis-version=redis_7_0 \
    --tier=basic \
    --network=default

# Capture Redis Host/Port
REDIS_HOST=$(gcloud redis instances describe $REDIS_NAME --region=$REGION --format="value(host)")
REDIS_PORT=$(gcloud redis instances describe $REDIS_NAME --region=$REGION --format="value(port)")
echo "Redis instance created at $REDIS_HOST:$REDIS_PORT"

# 4. Artifact Registry
echo "Creating Artifact Registry..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="DPR-RC Container Repository"

# 5. IAM Setup
echo "Creating Service Account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT \
    --display-name="DPR RC Agent Service Account"

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
    --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/dpr-agent:latest \\
    --region=${REGION} \\
    --service-account=${SA_EMAIL} \\
    --set-env-vars="REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},LOG_BUCKET=${BUCKET_NAME},ROLE=active" \\
    --allow-unauthenticated

# Deploy Passive Worker (Scale to 3 minimum for consensus)
gcloud run deploy dpr-passive-worker \\
    --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/dpr-agent:latest \\
    --region=${REGION} \\
    --service-account=${SA_EMAIL} \\
    --set-env-vars="REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},LOG_BUCKET=${BUCKET_NAME},ROLE=passive,HISTORY_BUCKET=${HISTORY_BUCKET_NAME}" \\
    --min-instances=3 \\
    --no-allow-unauthenticated
EOF

chmod +x deploy_commands.sh
echo "Infrastructure provisioning complete. Run './deploy_commands.sh' after building the container."
