#!/bin/bash
if [ "$ROLE" = "active" ]; then
    echo "Starting Active Agent Controller..."
    exec uvicorn dpr_rc.active_agent:app --host 0.0.0.0 --port ${PORT:-8080}
elif [ "$ROLE" = "passive" ]; then
    echo "Starting Passive Agent Worker..."
    exec uvicorn dpr_rc.passive_agent:app --host 0.0.0.0 --port ${PORT:-8080}
elif [ "$ROLE" = "baseline" ]; then
    echo "Starting Baseline RAG Agent (for comparison)..."
    exec uvicorn dpr_rc.baseline_agent:app --host 0.0.0.0 --port ${PORT:-8080}
else
    echo "Role not specified or recognized. Defaulting to benchmark mode."
    exec python -m benchmark.benchmark
fi
