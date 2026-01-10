# DPR-RC: Distributed Passive Retrieval with Resonant Consensus

Core implementation of the DPR-RC system for multi-perspective information retrieval with consensus-based answer synthesis.

## Overview

DPR-RC implements the Resonant Consensus Protocol (RCP v4) which:

1. **L1 Routing** - Routes queries to temporal shards based on timestamp context
2. **L2 Verification** - Verifies retrieved content via SLM semantic matching
3. **L3 Topology** - Calculates semantic quadrant coordinates for each response
4. **Consensus** - Synthesizes answers through resonant consensus from multiple workers

## Configuration

All system parameters are externalized to `dpr_rc/config/dpr_rc_config.yaml`:

### Config Sections

| Section | Description |
|---------|-------------|
| `slm` | Model names, generation settings, service config |
| `embedding` | Model name, dimension, word limit |
| `consensus` | RCP v4 thresholds (theta, tau, vote_threshold, etc.) |
| `verification` | Content limits, weights, fallback threshold |
| `quadrant` | Coordinate weights and precision |
| `query` | Results per shard, batch size, snippet length |
| `worker` | Default ID, cluster, epoch, scale, port |
| `logging` | Payload logging settings |
| `epoch` | Recent epoch start date, default shard |

### Configuration API

```python
from dpr_rc.config import get_dpr_config

config = get_dpr_config()

# Access nested config
slm_model = config['slm']['models']['default']
consensus_threshold = config['consensus']['mean_threshold']
```

### Environment Variable Overrides

Most config values can be overridden via environment variables:

```bash
SLM_MODEL=Qwen/Qwen2-0.5B-Instruct
SLM_SERVICE_URL=http://localhost:8081
WORKER_ID=worker-1
CLUSTER_ID=cluster-alpha
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Directory Structure

```
dpr_rc/
├── config/
│   ├── __init__.py           # Config loader (get_dpr_config)
│   └── dpr_rc_config.yaml    # All DPR-RC settings
├── application/
│   ├── dtos/                 # Data Transfer Objects
│   ├── interfaces/           # Service interfaces
│   └── use_cases/
│       └── process_query_use_case.py  # Main query processing
├── domain/
│   ├── passive_agent/
│   │   ├── entities/         # Domain entities
│   │   ├── repositories/     # Repository interfaces
│   │   └── services/
│   │       ├── verification_service.py  # L2 SLM verification
│   │       ├── quadrant_service.py      # L3 topology calculation
│   │       └── rfi_processor.py         # RFI orchestration
│   └── slm/
│       └── services/         # SLM domain services
├── infrastructure/
│   ├── passive_agent/
│   │   ├── clients/          # HTTP/Direct SLM clients
│   │   ├── repositories/
│   │   │   └── chromadb_repository.py  # Vector storage
│   │   └── factory.py        # Dependency injection
│   └── slm/
│       ├── backends/         # Transformers backend
│       └── factory.py        # SLM factory
├── passive_agent.py          # FastAPI passive worker
├── models.py                 # Shared Pydantic models
├── embedding_utils.py        # Embedding utilities
└── logging_utils.py          # Structured logging
```

## Key Parameters (RCP v4)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | 0.5 | Cluster approval threshold (Eq. 1) |
| `tau` | 0.6 | Consensus threshold (Eq. 4) |
| `vote_threshold` | 0.5 | Binary vote confidence threshold |
| `mean_threshold` | 0.7 | High consensus score threshold |
| `std_threshold` | 0.2 | Low std deviation for consensus |
| `asymmetric_threshold` | 0.4 | Threshold for asymmetric quadrant |
| `dissonant_threshold` | 0.3 | Threshold for dissonant polarization |

## Running the Passive Worker

```bash
# Default configuration
python -m dpr_rc.passive_agent

# With environment overrides
SLM_SERVICE_URL=http://localhost:8081 \
WORKER_ID=worker-1 \
python -m dpr_rc.passive_agent
```

## Data Flow (3-Layer Pipeline)

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ L1 ROUTING                                              │
│ Query → Time-based shard selection → Target shards      │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ L2 VERIFICATION (per worker)                            │
│ ChromaDB retrieval → SLM semantic verify → Confidence   │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ L3 CONSENSUS                                            │
│ Quadrant calculation → Vote aggregation → Superposition │
└─────────────────────────────────────────────────────────┘
  │
  ▼
Final Answer (consensus or perspectival)
```

## Dependencies

- ChromaDB for vector storage
- Sentence Transformers for embeddings (all-MiniLM-L6-v2)
- Transformers for SLM inference (Qwen2-0.5B-Instruct)
- FastAPI for worker HTTP interface
