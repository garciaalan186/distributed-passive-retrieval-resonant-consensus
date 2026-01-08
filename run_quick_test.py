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
# Model Configuration - Mixed Model Mode
# - Phi-3-mini (4-bit) for verification (accurate)
# - Qwen2-0.5B for enhancement (fast)
os.environ["SLM_MODEL"] = "microsoft/Phi-3-mini-4k-instruct"  # Main model for verification
os.environ["SLM_USE_4BIT_QUANTIZATION"] = "true"  # ~2GB per instance
os.environ["SLM_FAST_MODEL"] = "Qwen/Qwen2-0.5B-Instruct"  # Fast model for enhancement
os.environ["SLM_FAST_MAX_TOKENS"] = "100"  # Enhancement needs fewer tokens
os.environ["SLM_TIMEOUT"] = "45"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["LOCAL_DATASET_PATH"] = "benchmark_results_local/mini/dataset.json"
os.environ["HISTORY_SCALE"] = "mini"
os.environ["LOG_LEVEL"] = "INFO"  # Less verbose for quick test
os.environ["CHROMA_DB_PATH"] = "benchmark_results_local/chroma_db"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["SLM_MAX_TOKENS"] = "150"  # Reduced from 500 - verification responses are short
os.environ["SLM_ATTN_IMPL"] = "sdpa"  # Use PyTorch native attention (faster for small models)

# Enable query enhancement with fast model
os.environ["ENABLE_QUERY_ENHANCEMENT"] = "true"
os.environ["CONTROLLER_URL"] = "http://localhost:8080"
os.environ["SLM_SERVICE_URL"] = "http://localhost:8081"
os.environ["PASSIVE_WORKER_URL"] = "http://localhost:8082"

# Tier 1 Optimization: Parallel query execution
# ENABLED: Process-based parallelism avoids asyncio overhead
os.environ["ENABLE_PARALLEL_QUERIES"] = "true"
os.environ["MAX_CONCURRENT_QUERIES"] = "4"  # 4 parallel processes

# Tier 3B Optimization: Multi-GPU parallel processing
# ENABLED: Each process gets its own GPU via CUDA_VISIBLE_DEVICES
# With 2 GPUs and Qwen2-0.5B (~0.5GB), we can run 4 workers (2 per GPU)
os.environ["ENABLE_MULTI_GPU_WORKERS"] = "true"
os.environ["NUM_WORKER_THREADS"] = "4"  # 4 GPU workers total
os.environ["NUM_GPUS"] = "2"  # Actual number of GPUs for round-robin

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
