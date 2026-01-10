#!/bin/bash
# Note: Cloud deployment roles (active, slm) have been removed.
# Use benchmark mode for local testing.
if [ "$ROLE" = "passive" ]; then
    echo "Starting Passive Agent Worker..."
    exec uvicorn dpr_rc.passive_agent:app --host 0.0.0.0 --port ${PORT:-8080}
else
    echo "Running benchmark mode..."
    exec python run_benchmark.py ${BENCHMARK_SCALE:-small}
fi
