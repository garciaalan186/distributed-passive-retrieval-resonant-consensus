FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Qwen2-0.5B model to eliminate runtime download (saves 3-5min startup time)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    print('Pre-downloading Qwen2-0.5B-Instruct model...'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B-Instruct', torch_dtype='auto', trust_remote_code=True); \
    print('Model pre-download complete')"

# Copy application code
COPY dpr_rc/ /app/dpr_rc/
COPY benchmark/ /app/benchmark/
COPY scripts/ /app/scripts/

# Set env vars
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=false

# Entrypoint script to choose role
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

