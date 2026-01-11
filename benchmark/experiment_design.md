# Experimental Design: DPR-RC Scalability Validation

**Status:** Ready for Implementation

**Constraints:**
- Hardware: Multi-GPU / Cloud (1M+ events feasible)
- Time Budget: Multiple days (full suite)
- Metrics: Core + GPU/memory utilization

## Research Hypotheses

### H1: Accuracy Advantage at Scale
As historical context size increases, DPR-RC maintains higher accuracy than baseline RAG.
- **H1₀** (null): accuracy_dprrc ≤ accuracy_baseline
- **H1₁** (alternative): accuracy_dprrc > accuracy_baseline

### H2: Latency Scalability
DPR-RC latency grows sub-linearly with data scale while baseline grows linearly or worse.
- **H2₀** (null): latency_dprrc scales ≥ O(n)
- **H2₁** (alternative): latency_dprrc scales O(log n) or O(1)

### H3: Hallucination Resistance
DPR-RC hallucination rate remains near-zero regardless of scale, while baseline increases with scale.
- **H3₀** (null): hallucination_dprrc ≥ hallucination_baseline at all scales
- **H3₁** (alternative): hallucination_dprrc < hallucination_baseline, gap widens with scale

## Experimental Design

### Design Type
**2 × K Factorial Design** with repeated measures
- **Factor A**: System (DPR-RC vs Baseline) - 2 levels
- **Factor B**: Data Scale - K levels (7 proposed)
- **Replication**: N queries per condition (minimum 50 for statistical power)

### Independent Variables

#### Primary IV: Data Scale
Geometric progression to capture scaling behavior:

| Scale | Events | Est. Tokens | Temporal Span | Shards |
|-------|--------|-------------|---------------|--------|
| xs    | 100    | ~10K        | 2 years       | 2      |
| s     | 500    | ~50K        | 5 years       | 5      |
| m     | 2,500  | ~250K       | 10 years      | 10     |
| l     | 10,000 | ~1M         | 10 years      | 10     |
| xl    | 50,000 | ~5M         | 10 years      | 10     |
| xxl   | 200,000| ~20M        | 10 years      | 10     |
| xxxl  | 1,000,000| ~100M     | 10 years      | 10     |

#### Secondary IV: Query Type (stratified)
- `temporal_recall` - Single time point retrieval
- `consensus_detection` - Multi-source agreement
- `perspective_divergence` - Conflicting viewpoints
- `causal_chain` - Multi-hop temporal reasoning

### Dependent Variables

| Variable | Measurement | Unit |
|----------|-------------|------|
| **Accuracy** | Entity recall rate | % (0-100) |
| **Latency** | Per-query response time | ms |
| **Latency Breakdown** | L1/L2/L3 stage times | ms |
| **Hallucination Rate** | Forbidden term occurrence | % |
| **Grounding Rate** | Required term presence | % |
| **Throughput** | Queries per second | qps |

### Control Variables (Fixed)
- Random seed: 42
- Embedding model: all-MiniLM-L6-v2
- SLM model: Qwen/Qwen2-0.5B-Instruct
- Hardware: [Document GPU/CPU specs]
- ChromaDB settings: Default HNSW
- Consensus thresholds: From dpr_rc_config.yaml

## Data Collection Protocol

### Per-Query Record Schema
```json
{
  "experiment_id": "exp_2024_001",
  "run_id": "run_001",
  "scale": "xl",
  "system": "dprrc",
  "query_id": "temporal_Domain_2020",
  "query_type": "temporal_recall",
  "query_text": "...",
  "timestamp_context": "2020-12-31",

  "metrics": {
    "latency_total_ms": 1234.5,
    "latency_l1_routing_ms": 10.2,
    "latency_l2_retrieval_ms": 50.3,
    "latency_l2_verification_ms": 800.1,
    "latency_l3_consensus_ms": 374.0,

    "accuracy_entity_recall": 0.85,
    "accuracy_matched_count": 17,
    "accuracy_expected_count": 20,

    "hallucination_detected": false,
    "hallucination_terms_found": [],
    "grounding_passed": true,
    "grounding_terms_missing": []
  },

  "response": {
    "answer": "...",
    "confidence": 0.92,
    "sources": ["event_001", "event_002"],
    "consensus_count": 3,
    "perspectival_count": 1
  },

  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "dataset_events": 50000,
    "dataset_tokens_est": 5000000
  },

  "resource_usage": {
    "gpu_memory_allocated_mb": 4096,
    "gpu_memory_peak_mb": 5120,
    "gpu_utilization_pct": 85.5,
    "cpu_utilization_pct": 45.2,
    "ram_used_mb": 8192
  }
}
```

### Aggregate Statistics Record
```json
{
  "scale": "xl",
  "system": "dprrc",
  "n_queries": 100,

  "latency": {
    "mean": 1234.5,
    "std": 156.2,
    "median": 1200.0,
    "p5": 980.0,
    "p95": 1520.0,
    "min": 850.0,
    "max": 1800.0
  },

  "accuracy": {
    "mean": 0.92,
    "std": 0.08,
    "by_query_type": {
      "temporal_recall": 0.95,
      "consensus_detection": 0.88,
      "perspective_divergence": 0.85
    }
  },

  "hallucination": {
    "rate": 0.02,
    "count": 2,
    "total": 100
  },

  "resource_usage": {
    "gpu_memory_mean_mb": 4500,
    "gpu_memory_peak_mb": 6200,
    "gpu_utilization_mean_pct": 82.3,
    "cpu_utilization_mean_pct": 48.1,
    "ram_peak_mb": 12000
  }
}
```

## Implementation Plan

### New Files to Create

```
benchmark/
├── experiments/
│   ├── __init__.py
│   ├── experiment_runner.py      # Main orchestrator
│   ├── scale_generator.py        # Generate datasets at each scale
│   ├── metrics_collector.py      # Structured data collection
│   └── statistical_analysis.py   # Analysis utilities
├── config/
│   └── scales/
│       ├── xs.yaml    # 100 events
│       ├── s.yaml     # 500 events
│       ├── m.yaml     # 2,500 events
│       ├── l.yaml     # 10,000 events
│       ├── xl.yaml    # 50,000 events
│       ├── xxl.yaml   # 200,000 events
│       └── xxxl.yaml  # 1,000,000 events
└── results/
    └── experiment_{id}/
        ├── raw/           # Per-query JSON records
        ├── aggregated/    # Summary statistics
        ├── plots/         # Generated visualizations
        └── report.md      # Auto-generated summary
```

### Modifications to Existing Files

| File | Changes |
|------|---------|
| `benchmark/research_benchmark.py` | Add detailed timing instrumentation |
| `dpr_rc/application/use_cases/process_query_use_case.py` | Add per-stage timing |
| `benchmark/infrastructure/executors/dprrc_query_executor.py` | Capture stage breakdowns |
| `benchmark/config/benchmark_config.yaml` | Add experiment settings |

### Entry Point
```bash
# Run full experiment suite
python3 -m benchmark.experiments.experiment_runner \
  --scales xs,s,m,l,xl,xxl \
  --queries-per-scale 100 \
  --seed 42 \
  --output-dir benchmark/results/exp_001

# Run single scale for testing
python3 -m benchmark.experiments.experiment_runner \
  --scales m \
  --queries-per-scale 20 \
  --dry-run
```

## Statistical Analysis Plan

### 1. Descriptive Statistics
- Mean, SD, median, IQR for each metric
- Histograms and box plots by scale and system

### 2. Scaling Analysis (H2)
Fit regression models to latency vs. log(data_size):
- **Model 1**: O(1) - constant: `latency = β₀`
- **Model 2**: O(log n) - logarithmic: `latency = β₀ + β₁·log(n)`
- **Model 3**: O(n) - linear: `latency = β₀ + β₁·n`

Compare models using AIC/BIC. Report R² and residual analysis.

### 3. Hypothesis Testing
- **H1 (Accuracy)**: Two-way ANOVA (System × Scale), post-hoc Tukey HSD
- **H2 (Latency)**: Regression coefficient significance tests
- **H3 (Hallucination)**: Chi-square test per scale, logistic regression for trend

### 4. Effect Sizes
- Cohen's d for DPR-RC vs Baseline at each scale
- η² (eta-squared) for ANOVA effects

### 5. Confidence Intervals
- 95% CI for all point estimates
- Bootstrap CI for non-normal distributions

## Expected Output for Paper

### Tables
1. **Table 1**: Dataset characteristics by scale
2. **Table 2**: Accuracy comparison (mean ± SD) by scale and system
3. **Table 3**: Latency statistics by scale and system
4. **Table 4**: Hallucination rates with 95% CI
5. **Table 5**: Scaling regression coefficients

### Figures
1. **Figure 1**: Accuracy vs. Data Scale (line plot with error bars)
2. **Figure 2**: Latency vs. Data Scale (log-log plot with fitted curves)
3. **Figure 3**: Latency breakdown by pipeline stage (stacked bar)
4. **Figure 4**: Hallucination rate vs. Scale (with baseline comparison)
5. **Figure 5**: Query-type breakdown (heatmap)
6. **Figure 6**: GPU memory usage vs. Scale (resource efficiency)

### Statistical Reporting
All results reported with:
- Effect size (d or η²)
- 95% confidence intervals
- p-values (with Bonferroni correction for multiple comparisons)
- Sample sizes

## Verification Steps

1. **Pilot Run**: Execute xs and s scales to verify instrumentation
2. **Data Validation**: Check all required fields are captured
3. **Reproducibility**: Re-run with same seed, verify identical results
4. **Baseline Sanity**: Verify baseline accuracy degrades as expected at scale

## Resource Tracking Implementation

Use `pynvml` for GPU metrics and `psutil` for CPU/RAM:
```python
import pynvml
import psutil

# Initialize once at start
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Per-query sampling
def get_resource_snapshot():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_memory_allocated_mb": mem_info.used // (1024*1024),
        "gpu_utilization_pct": util.gpu,
        "cpu_utilization_pct": psutil.cpu_percent(),
        "ram_used_mb": psutil.virtual_memory().used // (1024*1024)
    }
```

## Estimated Runtime

| Scale | Events | Est. Time (DPR-RC) | Est. Time (Baseline) |
|-------|--------|-------------------|---------------------|
| xs    | 100    | ~5 min            | ~2 min              |
| s     | 500    | ~15 min           | ~5 min              |
| m     | 2,500  | ~1 hr             | ~20 min             |
| l     | 10,000 | ~4 hr             | ~1 hr               |
| xl    | 50,000 | ~12 hr            | ~4 hr               |
| xxl   | 200,000| ~24 hr            | ~10 hr              |
| xxxl  | 1M     | ~48 hr            | ~24 hr              |

**Total estimated runtime**: ~4-5 days for full suite with 100 queries per scale.
