#!/bin/bash
if [ "$ROLE" = "active" ]; then
    echo "Starting Active Agent Controller..."
    # Ingest foveated index for routing
    echo "Running Foveated Index Ingestion..."
    python scripts/ingest_foveated_index.py --scale ${BENCHMARK_SCALE:-medium}
    
    exec uvicorn dpr_rc.active_agent:app --host 0.0.0.0 --port ${PORT:-8080}
elif [ "$ROLE" = "passive" ]; then
    echo "Starting Passive Agent Worker..."
    exec uvicorn dpr_rc.passive_agent:app --host 0.0.0.0 --port ${PORT:-8080}
elif [ "$ROLE" = "slm" ]; then
    echo "Starting SLM Service (Qwen2-0.5B-Instruct)..."
    echo "Model: ${SLM_MODEL:-Qwen/Qwen2-0.5B-Instruct}"
    exec uvicorn dpr_rc.slm_service:app --host 0.0.0.0 --port ${SLM_PORT:-${PORT:-8081}}
else
    echo "Role not specified or recognized. Defaulting to benchmark mode."
    exec python -m benchmark.benchmark
fi
