FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We include torch and transformers for the local SLM, 
# and redis/chromadb for core logic.
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-logging \
    redis \
    chromadb \
    pydantic \
    numpy \
    transformers \
    torch \
    accelerate \
    fastapi \
    uvicorn \
    python-json-logger

# Copy application code
COPY dpr_rc/ /app/dpr_rc/
COPY benchmark/ /app/benchmark/

# Set env vars
ENV PYTHONUNBUFFERED=1

# Entrypoint script to choose role
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
if [ "\$ROLE" = "active" ]; then
    echo "Starting Active Agent Controller..."
    exec uvicorn dpr_rc.active_agent:app --host 0.0.0.0 --port 8080
elif [ "\$ROLE" = "passive" ]; then
    echo "Starting Passive Agent Worker..."
    exec python -m dpr_rc.passive_agent
else
    echo "Role not specified or recognized. Defaulting to benchmark mode."
    exec python -m benchmark.benchmark
fi
EOF

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
