# DPR-RC Benchmark System

Benchmark suite for evaluating the Distributed Passive Retrieval with Resonant Consensus (DPR-RC) system against baseline RAG retrieval.

## Quick Start

```bash
# Run a mini benchmark (fastest, for testing)
python3 run_benchmark.py mini

# Run other scales
python3 run_benchmark.py small
python3 run_benchmark.py medium
python3 run_benchmark.py large
python3 run_benchmark.py stress
```

## Configuration

The benchmark system uses a YAML-based configuration with two types of config files:

### Scale-Independent Config

`benchmark/config/benchmark_config.yaml` - Settings that don't change with benchmark scale:

| Section | Description |
|---------|-------------|
| `services` | Controller URL, worker URL, SLM service URL, timeouts |
| `evaluation` | Recall threshold, min response length, superposition settings |
| `hallucination` | Max retries, base delay, confidence threshold |
| `executor` | Max concurrent queries, worker threads, GPU settings |
| `synthetic` | Year range, seed, event type probabilities |

### Scale-Specific Configs

`benchmark/config/scales/{scale}.yaml` - Settings that vary by benchmark size:

| Scale | Events/Topic/Year | Domains | Perspectives | Use Case |
|-------|-------------------|---------|--------------|----------|
| mini | 2 | 1 | 2 | Quick testing |
| small | 10 | 2 | 2 | Local dev |
| medium | 25 | 3 | 3 | Integration |
| large | 50 | 4 | 3 | Performance |
| stress | 100 | 5 | 3 | Load testing |

## Directory Structure

```
benchmark/
├── config/
│   ├── __init__.py              # Config loader (get_config, get_scale_config)
│   ├── benchmark_config.yaml    # Scale-independent settings
│   └── scales/
│       ├── mini.yaml
│       ├── small.yaml
│       ├── medium.yaml
│       ├── large.yaml
│       └── stress.yaml
├── core/
│   └── hallucination_detector.py  # SLM-based hallucination detection
├── domain/
│   ├── interfaces/                # Abstract interfaces
│   └── services/
│       └── evaluation_service.py  # Correctness evaluation
├── infrastructure/
│   └── executors/
│       └── dprrc_query_executor.py  # Query execution via UseCase
├── synthetic/
│   └── generator.py               # Synthetic dataset generation
└── research_benchmark.py          # Main benchmark orchestrator
```

## Configuration API

```python
from benchmark.config import get_config, get_scale_config, get_all_scales

# Get scale-independent config
config = get_config()
timeout = config['services']['timeouts']['query']

# Get scale-specific config
mini_config = get_scale_config('mini')
events_per_year = mini_config['events_per_topic_per_year']

# Get all available scales
all_scales = get_all_scales()  # Dict keyed by scale name
```

## Environment Variable Overrides

Config values can be overridden via environment variables:

```bash
CONTROLLER_URL=http://localhost:8080
SLM_SERVICE_URL=http://localhost:8081
```

## Output

Results are saved to `benchmark_results_local/{scale}/`:

- `comparison.json` - Full benchmark results including:
  - Accuracy rates (DPR-RC vs baseline)
  - Hallucination rates
  - Latency metrics (mean, p95)
  - Per-query details

Example output:
```json
{
  "total_queries": 5,
  "dprrc_correct_rate": 1.0,
  "baseline_correct_rate": 1.0,
  "dprrc_hallucination_rate": 0.0,
  "baseline_hallucination_rate": 0.0,
  "dprrc_mean_latency": 26702.7,
  "baseline_mean_latency": 1375.1
}
```

## Synthetic Data

The benchmark uses phonotactic noun generation and alternate universe physics to create datasets with zero overlap with real-world knowledge, eliminating LLM parametric knowledge confounds.

Key component: `benchmark/synthetic/generator.py` (SyntheticHistoryGeneratorV2)
