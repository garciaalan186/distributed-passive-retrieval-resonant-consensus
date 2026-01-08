"""
Quick Test - Run 5 queries with hallucination detection
"""

import os
import sys
import json

# Ensure we can import from dpr_rc and benchmark modules
sys.path.append(os.getcwd())

# Set Environment Variables for Local Execution
os.environ["USE_NEW_EXECUTOR"] = "true"
os.environ["USE_DIRECT_SERVICES"] = "true"
os.environ["BENCHMARK_SCALE"] = "mini"
os.environ["SLM_MODEL"] = "microsoft/Phi-3-mini-4k-instruct"  # Full-size model (Qwen2 has 0% accuracy)
os.environ["SLM_TIMEOUT"] = "60"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["LOCAL_DATASET_PATH"] = "benchmark_results_local/mini/dataset.json"
os.environ["HISTORY_SCALE"] = "mini"
os.environ["LOG_LEVEL"] = "INFO"  # Less verbose for quick test
os.environ["CHROMA_DB_PATH"] = "benchmark_results_local/chroma_db"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["SLM_MAX_TOKENS"] = "500"
os.environ["SLM_ATTN_IMPL"] = "flash_attention_2"
os.environ["CONTROLLER_URL"] = "http://localhost:8080"
os.environ["SLM_SERVICE_URL"] = "http://localhost:8081"
os.environ["PASSIVE_WORKER_URL"] = "http://localhost:8082"

# Tier 1 Optimization: Parallel query execution
# DISABLED: Will be re-enabled after Tier 3B is validated
os.environ["ENABLE_PARALLEL_QUERIES"] = "false"
os.environ["MAX_CONCURRENT_QUERIES"] = "6"  # Will be used once Tier 3B is validated

# Tier 3B Optimization: Multi-GPU parallel shard processing with ThreadPoolExecutor
# ENABLED: Baseline accuracy verified, testing 2x speedup with dual GPU
os.environ["ENABLE_MULTI_GPU_WORKERS"] = "false"  # Disabled - Qwen2 gives 0% accuracy, Phi-3 too large
os.environ["NUM_WORKER_THREADS"] = "1"  # Sequential processing with Phi-3

from benchmark.research_benchmark import ResearchBenchmarkSuite
from pathlib import Path

def main():
    print("=== DPR-RC Quick Test (5 Queries) ===\n")

    # Load existing dataset
    dataset_path = Path("benchmark_results_local/mini/dataset.json")
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run the full benchmark first to generate the dataset.")
        return

    with open(dataset_path) as f:
        dataset = json.load(f)

    glossary_path = Path("benchmark_results_local/mini/glossary.json")
    with open(glossary_path) as f:
        glossary = json.load(f)

    # Take only first 5 queries
    queries = dataset['queries'][:5]
    print(f"Running {len(queries)} queries...\n")

    benchmark = ResearchBenchmarkSuite(output_dir="benchmark_results_local")

    # Ingest dataset
    print("Ingesting dataset into ChromaDB...")
    benchmark._ingest_dataset(dataset)

    # Run DPR-RC queries
    print("\nRunning DPR-RC queries...")
    results_dir = Path("benchmark_results_local/mini/dprrc_results")
    dprrc_results = benchmark.run_dprrc_queries(queries, results_dir)

    # Run baseline (optional, for comparison)
    print("\nRunning baseline queries...")
    baseline_dir = Path("benchmark_results_local/mini/baseline_results")
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
    comparison_path = Path("benchmark_results_local/mini/comparison_quick_test.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print results
    print("\n" + "="*60)
    print("QUICK TEST RESULTS")
    print("="*60)
    print(f"Total Queries: {comparison['total_queries']}")
    print(f"\nDPR-RC:")
    print(f"  Accuracy: {comparison['dprrc_correct_rate']*100:.1f}%")
    print(f"  Correct: {comparison['dprrc_correct_count']}/{comparison['total_queries']}")
    print(f"  Hallucination Rate: {comparison['dprrc_hallucination_rate']*100:.1f}%")
    print(f"  Hallucinations: {comparison['dprrc_hallucination_count']}/{comparison['total_queries']}")
    print(f"  Mean Latency: {comparison['dprrc_mean_latency']/1000:.1f}s")

    print(f"\nBaseline:")
    print(f"  Accuracy: {comparison['baseline_correct_rate']*100:.1f}%")
    print(f"  Hallucination Rate: {comparison['baseline_hallucination_rate']*100:.1f}%")

    print(f"\nResults saved to: {comparison_path}")

    # Show sample of hallucination details if any
    if comparison['dprrc_hallucination_count'] > 0:
        print("\n" + "="*60)
        print("HALLUCINATION DETAILS (Sample)")
        print("="*60)
        for detail in comparison['dprrc_hallucination_details'][:3]:
            print(f"\nQuery: {detail['query_id']}")
            print(f"  Type: {detail['type']}")
            print(f"  Severity: {detail['severity']}")
            print(f"  Explanation: {detail['explanation'][:100]}...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
