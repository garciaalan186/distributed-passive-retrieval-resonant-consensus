"""
Local Benchmark Runner

Mimics run_cloud_benchmark.sh but executes efficiently on local hardware.
Uses 'UseCase' mode for direct python execution (no HTTP overhead, easy debugging).
"""

import os
import sys

# Ensure we can import from dpr_rc and benchmark modules
sys.path.append(os.getcwd())

# Set Environment Variables for Local Execution
os.environ["USE_NEW_EXECUTOR"] = "true"  # Use direct python execution
os.environ["USE_DIRECT_SERVICES"] = "true" # Use in-process services (no HTTP)
os.environ["BENCHMARK_SCALE"] = "small"  # Start small
os.environ["SLM_MODEL"] = "microsoft/Phi-3-mini-4k-instruct" # Optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["LOCAL_DATASET_PATH"] = "benchmark_results_local/small/dataset.json" # Explicit path to data
os.environ["HISTORY_SCALE"] = "small"    # Ensure worker uses same scale
os.environ["LOG_LEVEL"] = "DEBUG" # Enable debug logging

# Configure URLs to dummy values (we are using direct execution, but some services might check)
os.environ["CONTROLLER_URL"] = "http://localhost:8080"
os.environ["SLM_SERVICE_URL"] = "http://localhost:8081"
os.environ["PASSIVE_WORKER_URL"] = "http://localhost:8082"

from benchmark.research_benchmark import ResearchBenchmarkSuite

def main():
    print("=== DPR-RC Local Benchmark (Optimized) ===")
    print(f"Scale: {os.environ['BENCHMARK_SCALE']}")
    print(f"Mode: UseCase (Direct Python Execution)")
    print(f"SLM Model: {os.environ['SLM_MODEL']}")
    
    benchmark = ResearchBenchmarkSuite(output_dir="benchmark_results_local")
    
    # Run the benchmark
    try:
        results = benchmark.run_full_benchmark()
        print("\nBenchmark completed successfully.")
        print(f"Results saved to: {benchmark.output_dir}")
    except Exception as e:
        print(f"\nFATAL ERROR during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
