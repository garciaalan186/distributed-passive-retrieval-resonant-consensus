# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Architecture

**Distributed Passive Retrieval with Resonant Consensus (DPR-RC)** is a multi-agent RAG system that implements time-sharded historical retrieval with consensus-based verification.

### Core Components

1. **ProcessQueryUseCase** (`dpr_rc/application/use_cases/process_query_use_case.py`)
   - Main query processing pipeline
   - L1 routing: Time-based shard selection
   - Query enhancement via SLM
   - L3 consensus calculation from worker votes

2. **Passive Agent Workers** (`dpr_rc/passive_agent.py`)
   - Lazy-loading architecture: Empty vector stores at startup
   - On-demand shard loading when RFI arrives
   - Pre-computed embeddings (no runtime embedding)
   - L2 verification: SLM-based semantic verification
   - L3 quadrant calculation: Maps responses to topological coordinates

3. **SLM Infrastructure** (`dpr_rc/infrastructure/slm/`)
   - Small language model for query enhancement and verification
   - Default: Qwen/Qwen2-0.5B-Instruct
   - Direct in-process execution via `DirectSLMService`

4. **Embedding System** (`dpr_rc/embedding_utils.py`)
   - Model-versioned pre-computed embeddings
   - Supports retroactive re-embedding with new models

### Data Flow (3-Layer Pipeline)

**L1 (Routing):** Query → Time-based shard selection → Target shards identified

**L2 (Verification):** Workers retrieve from ChromaDB → SLM verifies semantic match → Calculate confidence

**L3 (Consensus):** Workers compute semantic quadrant → Vote on response → Aggregate → Consensus or superposition

## Development Commands

### Running Local Benchmark

```bash
# Primary benchmark entry point
python3 run_benchmark.py --scale mini --max-queries 10

# Available scales: mini, small, medium, large
python3 run_benchmark.py --scale small

# Run with baseline comparison
python3 run_benchmark.py --scale mini --baseline
```

### Running Tests

```bash
# Run test suite
./run_tests.sh

# Or directly with pytest
pytest tests/ -v
```

### Data Generation

```bash
# Generate synthetic data + embeddings
python -m benchmark.generate_and_upload --scale medium --model all-MiniLM-L6-v2

# Retroactive re-embedding with new model
python -m benchmark.generate_and_upload --retroactive-embed --scale medium --model text-embedding-3-small

# List available data
python -m benchmark.generate_and_upload --list
```

## Key Architectural Patterns

### Lazy Loading Pattern
Workers start with empty vector stores. Shards are loaded on-demand when RFI specifies `target_shards`. Loaded shards are cached locally. This enables:
- Fast startup (no pre-loading)
- Low memory footprint (only accessed shards)
- Stateless workers

### Shard-Agnostic Workers
Workers can handle any shard, not tied to specific epochs. The L1 router determines target shards based on temporal context, workers load them dynamically.

### Pre-Computed Embeddings
Embeddings are computed once during data generation. Workers load `.npz` files containing vectors + metadata. Query embeddings are computed at runtime using the same model.

### Direct Service Execution
The benchmark uses direct Python execution via:
- `DirectSLMService` - In-process SLM inference
- `DirectWorkerService` - In-process worker calls
- `SimpleRouterService` - Tempo-normalized routing

This avoids HTTP/Redis overhead for local testing.

## Environment Variables

### Data Configuration
- `LOCAL_DATASET_PATH`: Path to local dataset JSON file
- `EMBEDDING_MODEL`: Model for embeddings (default: all-MiniLM-L6-v2)

### SLM Configuration
- `SLM_MODEL`: Model ID for verification (default: Qwen/Qwen2-0.5B-Instruct)
- `SLM_FAST_MODEL`: Optional fast model for query enhancement

### Debug
- `DEBUG_BREAKPOINTS=true`: Enable step-by-step pipeline logging
- `DEBUG_PAUSE_SECONDS=2`: Pause between debug breakpoints

### Benchmark
- `USE_DIRECT_SERVICES=true`: Use in-process services (default for local)
- `USE_NEW_EXECUTOR=true`: Use new benchmark executor

## Synthetic Data Generation

Uses **phonotactic noun generator** and **alternate universe physics** to create datasets with zero overlap with real-world knowledge. This eliminates LLM parametric knowledge confounds in evaluation.

**Key files:**
- `benchmark/synthetic_history.py`: Generator implementation
- `benchmark/generate_and_upload.py`: Data preparation

## Mathematical Model Reference

The system implements equations from the DPR-RC specification:

**L2 Verification (Eq. 9):**
```
C(r_p) = V(q, context_p) × 1/(1+i)
```
Where V is SLM semantic verification score, i is hierarchy depth.

**L3 Semantic Quadrant:**
Maps responses to 2D topological space `<v+, v->` based on consensus alignment.

## File Structure

- `dpr_rc/`: Core system components
  - `application/`: Use cases and DTOs
  - `domain/`: Business logic and entities
  - `infrastructure/`: Service implementations
  - `passive_agent.py`: Passive worker entry point
- `benchmark/`: Synthetic data generation and benchmarking
- `tests/`: Unit and integration tests
- `run_benchmark.py`: Primary benchmark entry point
- `requirements.txt`: Python dependencies
