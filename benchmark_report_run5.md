# DPR-RC Research Benchmark Report

**Generated**: 2025-12-20 21:16:05

## Executive Summary

This report presents empirical evaluation of Distributed Passive Retrieval with Resonant Consensus (DPR-RC) against baseline RAG architectures.

## Methodology

- **Dataset**: Phonotactic synthetic history (zero real-world term overlap)
- **Evaluation**: Superposition-aware correctness (correct answer present)
- **Hallucination Detection**: Terms not in generated glossary
- **Scale Levels**: Progressive scaling to find failure thresholds

## Results by Scale

| Scale | Events | Queries | DPR-RC Accuracy | Baseline Accuracy | DPR-RC Halluc. | Baseline Halluc. |
|-------|--------|---------|-----------------|-------------------|----------------|------------------|
| medium | 2475 | 66 | 0.00% | 0.00% | 0.00% | 0.00% |

## Latency Comparison

| Scale | DPR-RC Mean (ms) | Baseline Mean (ms) | Overhead |
|-------|------------------|--------------------|-----------|
| medium | 25805.0 | 0.0 | +0.0% |

## Key Findings

## Reproducibility

All artifacts for peer review are available in:
- `benchmark_results_research/`
  - `{scale}/dataset.json` - Generated datasets
  - `{scale}/glossary.json` - Phonotactic term definitions
  - `{scale}/dprrc_results/query_XXXX/` - Per-query audit trails
  - `{scale}/comparison.json` - Detailed evaluation metrics

## Audit Trail Structure

Each DPR-RC query includes:
- `input.json` - Query text and context
- `ground_truth.json` - Expected consensus/disputed claims
- `system_output.json` - Final response and confidence
- `audit_trail.json` - Complete L1/L2/L3 execution trace

