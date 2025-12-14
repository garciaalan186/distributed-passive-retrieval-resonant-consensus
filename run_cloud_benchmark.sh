#!/bin/bash
# run_cloud_benchmark.sh
# End-to-end test: Build -> Deploy -> Benchmark

set -e

RESULTS_DIR="benchmark_results_cloud"
mkdir -p $RESULTS_DIR

echo "=== DPR-RC Cloud Benchmark Start ==="
PROJECT_ID=$(gcloud config get-value project)
echo "Project: $PROJECT_ID"

# 1. Build Container (Cloud Build)
echo "--- Step 1: Building Container ---"
# We need to make sure we are in the root
# The infrastructure script defines REPO_NAME="dpr-repo"
REPO_NAME="dpr-repo"
REGION="us-central1"
IMAGE_URI="gcr.io/${PROJECT_ID}/dpr-agent:latest"

# Ensure repo exists (idempotent infrastructure run)
echo "Ensuring infrastructure..."
./infrastructure.sh

echo "Submitting build to Cloud Build..."
gcloud builds submit --tag $IMAGE_URI .

# 2. Deploy Services
echo "--- Step 2: Deploying Services ---"
# infrastructure.sh generates deploy_commands.sh with correct env vars
chmod +x deploy_commands.sh
./deploy_commands.sh

# 3. Get Controller URL
echo "--- Step 3: Resolving Endpoint ---"
CONTROLLER_URL=$(gcloud run services describe dpr-active-controller --region=$REGION --format='value(status.url)')
echo "Controller URL: $CONTROLLER_URL"

# 5. Run Benchmark
# -----------------------------------------------------------------------------
echo "--- Step 4: Running Benchmark ---"
# Verify endpoint
echo "Targeting: ${CONTROLLER_URL}"

# Run research-grade benchmark suite
export CONTROLLER_URL="${CONTROLLER_URL}" # Ensure CONTROLLER_URL is exported for the benchmark script
python3 -m benchmark.research_benchmark

# 6. Report
# -----------------------------------------------------------------------------
echo "--- Step 5: Generating Report ---"
echo "Results available in $RESULTS_DIR/"
