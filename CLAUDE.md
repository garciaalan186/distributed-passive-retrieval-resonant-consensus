# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Architecture

**Distributed Passive Retrieval with Resonant Consensus (DPR-RC)** is a multi-agent RAG system that implements time-sharded historical retrieval with consensus-based verification.

### Core Components

1. **Active Agent Controller** (`dpr_rc/active_agent.py`)
   - L1 routing: Time-based shard selection
   - Query enhancement via SLM service
   - Consensus calculation (L3) from worker votes
   - Two communication modes:
     - Redis Pub/Sub (local development)
     - HTTP-based worker calls (Cloud Run deployment)

2. **Passive Agent Workers** (`dpr_rc/passive_agent.py`)
   - Lazy-loading architecture: Empty vector stores at startup
   - On-demand shard loading from GCS when RFI arrives
   - Pre-computed embeddings loaded from GCS (no runtime embedding)
   - L2 verification: SLM-based semantic verification
   - L3 quadrant calculation: Maps responses to topological coordinates

3. **SLM Service** (`dpr_rc/slm_service.py`)
   - Small language model for query enhancement and verification
   - Default: Qwen/Qwen2-0.5B-Instruct
   - Separate microservice to enable GPU/CPU optimization

4. **Embedding System** (`dpr_rc/embedding_utils.py`)
   - Model-versioned pre-computed embeddings
   - Supports retroactive re-embedding with new models
   - GCS storage structure:
     ```
     gs://{bucket}/
     ├── raw/{scale}/shards/shard_{year}.json       # Plain text
     └── embeddings/{model}/{scale}/shards/shard_{year}.npz  # Vectors
     ```

### Data Flow (3-Layer Pipeline)

**L1 (Routing):** Active Agent → Time-based shard selection → Target shards identified

**L2 (Verification):** Workers retrieve from ChromaDB → SLM verifies semantic match → Calculate confidence

**L3 (Consensus):** Workers compute semantic quadrant → Vote on response → Active Agent aggregates → Consensus or superposition

### Communication Modes

**Local Development (Redis):**
- Active Agent publishes RFI to Redis stream
- Workers consume from stream, vote via Pub/Sub
- Active Agent collects votes from Redis

**Cloud Deployment (HTTP):**
- Active Agent calls worker HTTP endpoint directly
- Worker returns vote synchronously
- No Redis dependency (Cloud Run friendly)
- Configured via `USE_HTTP_WORKERS=true`

## Development Commands

### Local Testing

```bash
# Run all tests
./run_tests.sh

# Run specific test suites
pytest tests/test_active_unit.py -v
pytest tests/test_passive_unit.py -v
pytest tests/test_integration_local.py -v

# Run single test
pytest tests/test_active_unit.py::test_route_logic -v
```

### Local Development (requires Redis)

```bash
# Start Redis
redis-server

# Terminal 1: SLM Service
python -m dpr_rc.slm_service

# Terminal 2: Passive Worker
ROLE=passive python -m dpr_rc.passive_agent

# Terminal 3: Active Controller
ROLE=active python -m dpr_rc.active_agent

# Terminal 4: Benchmark
python -m benchmark.research_benchmark
```

### Cloud Deployment & Benchmarking

```bash
# Full cloud pipeline (builds, deploys, benchmarks, cleans up)
./run_cloud_benchmark.sh

# With debug mode (step-by-step logging)
DEBUG_BREAKPOINTS=true ./run_cloud_benchmark.sh

# Specific scale level
BENCHMARK_SCALE=medium ./run_cloud_benchmark.sh

# Custom embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2 ./run_cloud_benchmark.sh

# View debug logs during execution
./view_debug_logs.sh
```

### Data Generation

```bash
# Generate synthetic data + embeddings (runs in Cloud Build)
python -m benchmark.generate_and_upload --scale medium --model all-MiniLM-L6-v2

# Retroactive re-embedding with new model
python -m benchmark.generate_and_upload --retroactive-embed --scale medium --model text-embedding-3-small

# List available data
python -m benchmark.generate_and_upload --list
```

## Key Architectural Patterns

### Lazy Loading Pattern
Workers start with empty vector stores. Shards are loaded on-demand when RFI specifies `target_shards`. Loaded shards are cached locally. This enables:
- Fast container startup (no pre-loading)
- Low memory footprint (only accessed shards)
- Horizontal scalability (workers are stateless)

### Shard-Agnostic Workers
Workers can handle any shard, not tied to specific epochs. The Active Agent's L1 router determines target shards based on temporal context, workers load them dynamically.

### Pre-Computed Embeddings
Embeddings are computed once during data generation and stored in GCS. Workers load `.npz` files containing vectors + metadata. Query embeddings are computed at runtime using the same model.

### Model Versioning
Embeddings are stored in model-specific folders: `embeddings/{model_id}/{scale}/shards/`. This allows:
- Testing different embedding models on same raw data
- A/B comparison without data regeneration
- Retroactive re-embedding when upgrading models

## Environment Variables

### Critical Configuration

**Data Storage:**
- `HISTORY_BUCKET`: GCS bucket for raw data and embeddings
- `HISTORY_SCALE`: Dataset scale (small/medium/large/stress)
- `EMBEDDING_MODEL`: Model for embeddings (default: all-MiniLM-L6-v2)

**Communication Mode:**
- `USE_HTTP_WORKERS=true`: HTTP-based workers (Cloud Run)
- `USE_HTTP_WORKERS=false`: Redis Pub/Sub (local dev)
- `PASSIVE_WORKER_URL`: Worker endpoint for HTTP mode
- `REDIS_HOST`, `REDIS_PORT`: Redis connection for Pub/Sub mode

**Services:**
- `SLM_SERVICE_URL`: URL for SLM microservice
- `ENABLE_QUERY_ENHANCEMENT=true`: Use SLM for query enhancement

**Debug:**
- `DEBUG_BREAKPOINTS=true`: Enable step-by-step pipeline logging
- `DEBUG_PAUSE_SECONDS=2`: Pause between debug breakpoints

**Role Selection (entrypoint.sh routing):**
- `ROLE=active`: Run Active Agent Controller
- `ROLE=passive`: Run Passive Agent Worker
- `ROLE=slm`: Run SLM Service

### Benchmark Configuration

- `BENCHMARK_SCALE`: Scale to run (small/medium/large/stress/all)
- `CONTROLLER_URL`: Active Agent endpoint for benchmark

### Enhanced Observability

**Full Payload Logging:**
- `ENABLE_FULL_PAYLOAD_LOGGING=true`: Enable complete message content logging (default: true)
- `MAX_PAYLOAD_SIZE_BYTES=100000`: Max payload size before truncation (default: 100KB)

**What's Logged:**
- All request/response payloads with trace correlation
- Internal message exchanges (ChromaDB queries, vote creation)
- SLM verification calls with full context
- Complete A* ↔ A_h conversation flows

## API Documentation

OpenAPI 3.0 specifications are available for all services:

**Interactive Documentation:**
- Active Controller: `https://dpr-active-controller-{project}.run.app/docs`
- Passive Worker: `https://dpr-passive-worker-{project}.run.app/docs`
- SLM Service: `https://dpr-slm-service-{project}.run.app/docs`

**Specification Files:**
- `openapi/openapi_active_controller.yaml` - Query processing and consensus
- `openapi/openapi_passive_worker.yaml` - RFI processing and voting
- `openapi/openapi_slm_service.yaml` - Verification and enhancement

**Key Endpoints:**

*Active Controller:*
- `POST /query` - Main query processing (L1→L2→L3 pipeline)
- `GET /health` - Health check

*Passive Worker:*
- `POST /process_rfi` - Process Request For Information
- `GET /health` - Worker status and loaded shards

*SLM Service:*
- `POST /verify` - L2 semantic verification
- `POST /enhance_query` - Query enhancement
- `POST /check_hallucination` - Hallucination detection

## Enhanced Observability

### Full Message Logging

All services log complete message exchanges with trace correlation. Each log entry includes:
- **Request payload**: Full incoming request
- **Response payload**: Complete response data
- **Trace ID**: For cross-service correlation
- **Message type**: Classification (e.g., `client_query`, `slm_verify`, `worker_rfi`)
- **Metadata**: Timing metrics, service info

**Logged Exchange Points:**

*Active Controller (6 points):*
1. Client query receipt
2. SLM enhancement request/response
3. Worker HTTP request/response
4. Final client response

*Passive Worker (7 points):*
1. RFI receipt
2. ChromaDB query and results
3. SLM verification request/response
4. Vote creation
5. Vote response

*SLM Service (7 points):*
1. Model loading metrics
2. `/verify` request/response
3. `/enhance_query` request/response
4. `/check_hallucination` request/response

### Querying Logs

**Find all logs for a specific query:**
```bash
gcloud logging read "jsonPayload.trace_id=\"your-trace-id\"" \
  --project geometric-mnemic-manifolds-bm \
  --format json \
  --order asc
```

**Filter by message type:**
```bash
gcloud logging read "jsonPayload.message_type=\"slm_verify\"" \
  --limit 100 \
  --format json
```

**Find errors:**
```bash
gcloud logging read "severity>=ERROR" \
  --limit 50 \
  --format json
```

### Exchange History Download

Download complete message exchange sequences for any trace_id:

**Usage:**
```bash
# View in terminal (markdown format)
python scripts/download_query_history.py <trace_id>

# Save as JSON for analysis
python scripts/download_query_history.py <trace_id> --format json --output trace.json

# Save as markdown report
python scripts/download_query_history.py <trace_id> --output trace.md
```

**Features:**
- Chronological ordering of all message exchanges
- Filters for relevant message types (requests, responses, internal)
- Complete A* ↔ A_h conversation reconstruction
- Automatic integration with benchmark (saves `exchange_history.json` per query)

**Benchmark Integration:**

The benchmark automatically downloads exchange history for each query. Results directories contain:
- `system_output.json` - Final response from DPR-RC
- `ground_truth.json` - Expected answer
- `audit_trail.json` - Raw Cloud Logging entries
- `exchange_history.json` - Structured message flow (**NEW**)

### Disabling Full Payload Logging

For privacy or performance reasons, disable full payload logging:

```bash
export ENABLE_FULL_PAYLOAD_LOGGING=false
```

This falls back to hash-only logging (MD5 of payloads) while maintaining trace correlation.

## Testing Notes

### Test Dependencies
Tests use `fakeredis` for Redis mocking and `pytest` for test execution. ChromaDB may fail in restricted network environments where it cannot download embedding models - affected tests will skip gracefully.

### Integration Test Patterns
- `test_integration_local.py`: Full RFI → Vote → Consensus flow
- Uses shared `fakeredis` instance patched into both modules
- Worker runs in thread with controlled iteration count

### Unit Test Patterns
- `test_active_unit.py`: Routing logic, consensus calculation
- `test_passive_unit.py`: Retrieval, verification, voting (mocked Redis)

## Synthetic Data Generation

Uses **phonotactic noun generator** and **alternate universe physics** to create datasets with zero overlap with real-world knowledge. This eliminates LLM parametric knowledge confounds in evaluation.

**Key files:**
- `benchmark/synthetic_history.py`: Generator implementation
- `benchmark/generate_and_upload.py`: Cloud data preparation
- Output: `raw/` and `embeddings/` in GCS

## Debug Mode

When `DEBUG_BREAKPOINTS=true`, the system logs detailed output at each pipeline step:

1. Query Received (Client → Controller)
2. Query Enhancement (Controller → SLM)
3. L1 Routing (Controller → Router)
4. HTTP Worker Call (Controller → Worker)
5. Shard Loading (Worker → GCS)
6. Document Retrieval (Worker → ChromaDB)
7. L2 Verification (Worker → SLM)
8. L3 Quadrant Calculation
9. Vote Creation (Worker → Controller)
10. Consensus Calculation
11. Final Response (Controller → Client)

View logs with: `./view_debug_logs.sh`

## Cloud Run Deployment Architecture

**Single Container, Multiple Roles:** Same Docker image (`gcr.io/{project}/dpr-agent:latest`) deployed as three services, differentiated by `ROLE` env var:
- `dpr-active-controller` (ROLE=active)
- `dpr-passive-worker` (ROLE=passive)
- `dpr-slm-service` (ROLE=slm)

**HTTP Communication:** Active Controller calls workers via HTTP, workers return votes synchronously. No Redis dependency.

**Lazy Loading:** Workers load shards from GCS on first request, cache locally for container lifetime.

**Auto-Scaling:** Cloud Run scales workers based on request volume. Each worker can handle any shard.

## Mathematical Model Reference

The system implements equations from the DPR-RC specification:

**L2 Verification (Eq. 9):**
```
C(r_p) = V(q, context_p) × 1/(1+i)
```
Where V is SLM semantic verification score, i is hierarchy depth.

**L3 Semantic Quadrant:**
Maps responses to 2D topological space `<v+, v->` based on consensus alignment.

## Common Workflows

### Testing New Embedding Model
```bash
# 1. Generate embeddings with new model (in cloud)
EMBEDDING_MODEL=text-embedding-3-small ./run_cloud_benchmark.sh

# 2. Benchmark will automatically use new embeddings
# 3. Compare results against baseline in benchmark_results_research/
```

### Debugging Failed Queries
```bash
# 1. Enable debug mode
DEBUG_BREAKPOINTS=true ./run_cloud_benchmark.sh

# 2. In another terminal, tail logs
./view_debug_logs.sh

# 3. Check audit trails in benchmark_results_research/{scale}/dprrc_results/query_*/
```

### Local Development Iteration
```bash
# 1. Start services locally (requires Redis)
redis-server &
python -m dpr_rc.slm_service &
ROLE=passive python -m dpr_rc.passive_agent &
ROLE=active USE_HTTP_WORKERS=false python -m dpr_rc.active_agent

# 2. Test changes
pytest tests/ -v

# 3. Run local benchmark
CONTROLLER_URL=http://localhost:8080 python -m benchmark.research_benchmark
```

## File Structure Key Points

- `dpr_rc/`: Core system components (agents, models, utilities)
- `benchmark/`: Synthetic data generation and benchmarking
- `tests/`: Unit and integration tests
- `benchmark_results_research/`: Benchmark output with audit trails
- `*.sh`: Deployment and execution scripts
- `requirements.txt`: Python dependencies (CPU-only PyTorch)
- `Dockerfile`: Multi-role container definition
- `entrypoint.sh`: Role-based service routing
