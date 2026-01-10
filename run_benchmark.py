#!/usr/bin/env python3
"""
DPR-RC Local Benchmark

Unified entry point for running benchmarks at any scale level.

Usage:
    python run_benchmark.py mini      # 5 queries, ~600 events, ~2 min
    python run_benchmark.py small     # 15 queries, ~1200 events, ~5 min
    python run_benchmark.py medium    # 25 queries, ~2250 events, ~10 min
    python run_benchmark.py large     # 25 queries, ~5000 events, ~15 min

Query caps are graduated by scale to test retrieval accuracy over increasing
amounts of historical data, not just more queries.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure imports work
sys.path.insert(0, os.getcwd())

# Scale configurations
SCALE_CONFIG = {
    "mini": {
        "query_cap": 5,
        "description": "Quick smoke test",
    },
    "small": {
        "query_cap": 15,
        "description": "Development iteration",
    },
    "medium": {
        "query_cap": 25,
        "description": "Standard benchmark",
    },
    "large": {
        "query_cap": 25,
        "description": "Stress test (more historical data)",
    },
}


def setup_environment(scale: str):
    """Configure environment for local benchmark execution."""

    # Core execution mode
    os.environ["USE_NEW_EXECUTOR"] = "true"
    os.environ["USE_DIRECT_SERVICES"] = "true"
    os.environ["BENCHMARK_SCALE"] = scale
    os.environ["HISTORY_SCALE"] = scale

    # Dataset paths
    os.environ["LOCAL_DATASET_PATH"] = f"benchmark_results_local/{scale}/dataset.json"
    os.environ["CHROMA_DB_PATH"] = f"benchmark_results_local/chroma_db_{scale}"

    # Model configuration - Mixed Model Mode
    os.environ["SLM_MODEL"] = "microsoft/Phi-3-mini-4k-instruct"
    os.environ["SLM_USE_4BIT_QUANTIZATION"] = "true"
    os.environ["SLM_FAST_MODEL"] = "Qwen/Qwen2-0.5B-Instruct"
    os.environ["SLM_FAST_MAX_TOKENS"] = "100"
    os.environ["SLM_MAX_TOKENS"] = "150"
    os.environ["SLM_TIMEOUT"] = "45"
    os.environ["SLM_ATTN_IMPL"] = "sdpa"

    # Parallel execution
    os.environ["ENABLE_PARALLEL_QUERIES"] = "true"
    os.environ["MAX_CONCURRENT_QUERIES"] = "4"
    os.environ["ENABLE_MULTI_GPU_WORKERS"] = "true"
    os.environ["NUM_WORKER_THREADS"] = "4"
    os.environ["NUM_GPUS"] = "2"

    # Misc
    os.environ["ENABLE_QUERY_ENHANCEMENT"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Service URLs (not used in direct mode, but set for compatibility)
    os.environ["CONTROLLER_URL"] = "http://localhost:8080"
    os.environ["SLM_SERVICE_URL"] = "http://localhost:8081"
    os.environ["PASSIVE_WORKER_URL"] = "http://localhost:8082"


def run_benchmark(scale: str, generate_data: bool = False):
    """Run benchmark for specified scale."""

    config = SCALE_CONFIG[scale]
    query_cap = config["query_cap"]

    print(f"\n{'='*60}")
    print(f"DPR-RC Benchmark: {scale.upper()}")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")
    print(f"Query cap: {query_cap}")
    print()

    # Setup environment
    setup_environment(scale)

    # Import after environment setup
    from benchmark.research_benchmark import ResearchBenchmarkSuite
    from benchmark.synthetic import SyntheticHistoryGeneratorV2

    # Check for existing dataset or generate new one
    dataset_path = Path(f"benchmark_results_local/{scale}/dataset.json")
    glossary_path = Path(f"benchmark_results_local/{scale}/glossary.json")

    if not dataset_path.exists() or generate_data:
        print(f"Generating synthetic dataset for {scale}...")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        generator = SyntheticHistoryGeneratorV2()
        dataset = generator.generate(scale=scale)
        glossary = generator.get_glossary()

        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        with open(glossary_path, 'w') as f:
            json.dump(glossary, f, indent=2)

        print(f"Generated {len(dataset['events'])} events, {len(dataset['queries'])} queries")
    else:
        print(f"Loading existing dataset from {dataset_path}")
        with open(dataset_path) as f:
            dataset = json.load(f)
        with open(glossary_path) as f:
            glossary = json.load(f)

    # Apply query cap
    queries = dataset['queries'][:query_cap]
    print(f"Running {len(queries)} queries (capped from {len(dataset['queries'])} total)\n")

    # Initialize benchmark suite
    benchmark = ResearchBenchmarkSuite(output_dir="benchmark_results_local")

    # Ingest dataset into ChromaDB
    print("Ingesting dataset into ChromaDB...")
    benchmark._ingest_dataset(dataset)

    # Run DPR-RC queries
    print("\nRunning DPR-RC queries...")
    results_dir = Path(f"benchmark_results_local/{scale}/dprrc_results")
    dprrc_results = benchmark.run_dprrc_queries(queries, results_dir)

    # Run baseline (optional comparison)
    print("\nRunning baseline queries...")
    baseline_dir = Path(f"benchmark_results_local/{scale}/baseline_results")
    baseline_results = benchmark.run_baseline_queries(queries, baseline_dir)

    # Compare results with hallucination detection
    print("\nComparing results and detecting hallucinations...")
    comparison = benchmark.compare_results(
        queries,
        dprrc_results,
        baseline_results,
        glossary
    )

    # Save comparison
    comparison_path = Path(f"benchmark_results_local/{scale}/comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {scale.upper()}")
    print(f"{'='*60}")
    print(f"Total Queries: {len(queries)}")
    print()
    print("DPR-RC:")
    print(f"  Accuracy: {comparison['dprrc_correct_rate']*100:.1f}%")
    print(f"  Correct: {comparison['dprrc_correct_count']}/{len(queries)}")
    print(f"  Hallucination Rate: {comparison['dprrc_hallucination_rate']*100:.1f}%")
    print(f"  Hallucinations: {comparison['dprrc_hallucination_count']}/{len(queries)}")
    print(f"  Mean Latency: {comparison['dprrc_mean_latency']/1000:.1f}s")
    print()
    print("Baseline:")
    print(f"  Accuracy: {comparison['baseline_correct_rate']*100:.1f}%")
    print(f"  Hallucination Rate: {comparison['baseline_hallucination_rate']*100:.1f}%")
    print(f"  Mean Latency: {comparison['baseline_mean_latency']/1000:.1f}s")
    print()
    print(f"Results saved to: {comparison_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="DPR-RC Local Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_benchmark.py mini          # Quick 5-query test
    python run_benchmark.py medium        # Standard 25-query benchmark
    python run_benchmark.py large --regen # Regenerate dataset first

Scale Levels:
    mini   - 5 queries, ~600 events (smoke test)
    small  - 15 queries, ~1200 events (dev iteration)
    medium - 25 queries, ~2250 events (standard)
    large  - 25 queries, ~5000 events (stress test)
        """
    )

    parser.add_argument(
        "scale",
        choices=["mini", "small", "medium", "large"],
        help="Benchmark scale level"
    )

    parser.add_argument(
        "--regen", "--regenerate",
        action="store_true",
        dest="regenerate",
        help="Regenerate synthetic dataset before running"
    )

    args = parser.parse_args()

    # Validate scale
    if args.scale not in SCALE_CONFIG:
        print(f"Error: Unknown scale '{args.scale}'")
        print(f"Valid scales: {', '.join(SCALE_CONFIG.keys())}")
        sys.exit(1)

    # Run benchmark
    try:
        comparison = run_benchmark(args.scale, generate_data=args.regenerate)

        # Exit with appropriate code
        accuracy = comparison['dprrc_correct_rate']
        if accuracy >= 0.9:
            sys.exit(0)  # Success
        elif accuracy >= 0.7:
            sys.exit(0)  # Acceptable
        else:
            sys.exit(1)  # Below threshold

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
