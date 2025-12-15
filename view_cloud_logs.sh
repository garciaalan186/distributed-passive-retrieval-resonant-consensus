#!/bin/bash
# view_cloud_logs.sh
# View Cloud Build logs for DPR-RC benchmark runs

set -e

echo "=== DPR-RC Cloud Build Logs ==="
echo ""

# List recent builds
echo "--- Recent Builds ---"
gcloud builds list --limit=10 --format="table(id,status,createTime,duration)"

echo ""
echo "--- Latest Build Details ---"

# Get the most recent build ID
LATEST_BUILD=$(gcloud builds list --limit=1 --format="value(id)")

if [ -n "$LATEST_BUILD" ]; then
    echo "Build ID: $LATEST_BUILD"
    echo ""

    # Show build steps
    echo "--- Build Steps ---"
    gcloud builds describe "$LATEST_BUILD" --format="table(steps.id,steps.status,steps.timing.startTime)"

    echo ""
    echo "--- Full Logs (last 200 lines) ---"
    gcloud builds log "$LATEST_BUILD" 2>/dev/null | tail -200

    echo ""
    echo "=== To view complete logs: gcloud builds log $LATEST_BUILD ==="
else
    echo "No builds found"
fi
