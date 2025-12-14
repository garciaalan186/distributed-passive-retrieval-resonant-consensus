#!/bin/bash
# Deploy Active Controller
gcloud run deploy dpr-active-controller \
    --image=gcr.io/geometric-mnemic-manifolds-bm/dpr-agent:latest \
    --region=us-central1 \
    --service-account=dpr-agent-sa@geometric-mnemic-manifolds-bm.iam.gserviceaccount.com \
    --set-env-vars="REDIS_HOST=10.6.246.99,REDIS_PORT=6379,LOG_BUCKET=dpr-audit-logs-geometric-mnemic-manifolds-bm,ROLE=active,CONTROLLER_URL=http://localhost:8080/query" \
    --vpc-connector=dpr-vpc-connector \
    --vpc-egress=private-ranges-only \
    --allow-unauthenticated

# Deploy Passive Worker (Scale to 3 minimum for consensus)
gcloud run deploy dpr-passive-worker \
    --image=gcr.io/geometric-mnemic-manifolds-bm/dpr-agent:latest \
    --region=us-central1 \
    --service-account=dpr-agent-sa@geometric-mnemic-manifolds-bm.iam.gserviceaccount.com \
    --set-env-vars="REDIS_HOST=10.6.246.99,REDIS_PORT=6379,LOG_BUCKET=dpr-audit-logs-geometric-mnemic-manifolds-bm,ROLE=passive,HISTORY_BUCKET=dpr-history-data-geometric-mnemic-manifolds-bm" \
    --vpc-connector=dpr-vpc-connector \
    --vpc-egress=private-ranges-only \
    --min-instances=3 \
    --no-allow-unauthenticated
