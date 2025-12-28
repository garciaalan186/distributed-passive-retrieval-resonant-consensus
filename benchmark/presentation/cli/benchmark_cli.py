#!/usr/bin/env python3
"""
Benchmark CLI

Command-line interface for running DPR-RC benchmarks using the new clean architecture.

This CLI provides a simple way to run benchmarks with various configurations:
- Different scale levels (small, medium, large, stress)
- HTTP mode or UseCase mode (direct use case execution)
- Optional hallucination detection
- Configurable service URLs

Usage:
    # Run small benchmark with HTTP mode
    python -m benchmark.presentation.cli.benchmark_cli --scale small

    # Run with UseCase mode (no HTTP overhead)
    python -m benchmark.presentation.cli.benchmark_cli --scale small --use-new-executor

    # Run with custom URLs
    python -m benchmark.presentation.cli.benchmark_cli \\
        --scale medium \\
        --controller-url https://my-service.run.app \\
        --slm-service-url http://localhost:8081

    # Run without hallucination detection
    python -m benchmark.presentation.cli.benchmark_cli \\
        --scale small \\
        --no-hallucination-detection
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

from benchmark.application.dtos import RunBenchmarkRequest
from benchmark.infrastructure.factories import BenchmarkFactory


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DPR-RC Benchmark CLI - Run benchmarks with clean architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Scale configuration
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large", "stress"],
        default="small",
        help="Benchmark scale level (default: small)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="benchmark_results_new",
        help="Directory to save benchmark results (default: benchmark_results_new)"
    )

    # Execution mode
    parser.add_argument(
        "--use-new-executor",
        action="store_true",
        help="Use ProcessQueryUseCase directly instead of HTTP (for benchmark purity)"
    )

    # Service URLs
    parser.add_argument(
        "--controller-url",
        help="DPR-RC controller URL for HTTP mode (default: $CONTROLLER_URL or http://localhost:8080)"
    )

    parser.add_argument(
        "--worker-url",
        help="Worker URL for UseCase mode (default: $PASSIVE_WORKER_URL or http://localhost:8082)"
    )

    parser.add_argument(
        "--slm-service-url",
        help="SLM service URL for hallucination detection (default: $SLM_SERVICE_URL or http://localhost:8081)"
    )

    # Features
    parser.add_argument(
        "--no-hallucination-detection",
        action="store_true",
        help="Disable SLM-based hallucination detection (use rule-based fallback)"
    )

    parser.add_argument(
        "--baseline-mode",
        choices=["http", "local", "mock"],
        default="local",
        help="Baseline executor mode (default: local)"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible dataset generation"
    )

    # Timeout
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("DPR-RC Benchmark CLI")
    print("=" * 60)
    print(f"Scale: {args.scale}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Execution Mode: {'UseCase (direct)' if args.use_new_executor else 'HTTP'}")
    print(f"Baseline Mode: {args.baseline_mode}")
    print(f"Hallucination Detection: {'Enabled' if not args.no_hallucination_detection else 'Disabled (fallback)'}")
    if args.seed:
        print(f"Random Seed: {args.seed}")
    print("=" * 60)
    print()

    try:
        # Create factory
        print("Creating benchmark use case...")
        factory = BenchmarkFactory()
        use_case = factory.create_benchmark_use_case(
            scale=args.scale,
            output_dir=args.output_dir,
            use_new_executor=args.use_new_executor,
            controller_url=args.controller_url,
            worker_url=args.worker_url,
            slm_url=args.slm_service_url,
            baseline_mode=args.baseline_mode,
            timeout=args.timeout,
            enable_query_enhancement=True
        )

        # Create request
        slm_url = args.slm_service_url or os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
        request = RunBenchmarkRequest(
            scale=args.scale,
            output_dir=args.output_dir,
            enable_hallucination_detection=not args.no_hallucination_detection,
            slm_service_url=slm_url if not args.no_hallucination_detection else None,
            seed=args.seed
        )

        # Execute benchmark
        print("Running benchmark...")
        response = asyncio.run(use_case.execute(request))

        # Print results
        print()
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        if response.succeeded:
            print(f"Run ID: {response.run_id}")
            print(f"Scale: {response.scale}")
            print(f"Total Queries: {response.total_queries}")
            print()
            print("Accuracy:")
            print(f"  DPR-RC:   {response.dprrc_accuracy:.2%}")
            print(f"  Baseline: {response.baseline_accuracy:.2%}")
            print(f"  Improvement: {response.get_accuracy_improvement():+.1%}")
            print()
            print("Hallucination Rate:")
            print(f"  DPR-RC:   {response.dprrc_hallucination_rate:.2%}")
            print(f"  Baseline: {response.baseline_hallucination_rate:.2%}")
            print(f"  Reduction: {response.get_hallucination_reduction():+.1%}")
            print()
            print("Latency (ms):")
            print(f"  DPR-RC Mean:    {response.mean_latency_dprrc:.1f}")
            print(f"  Baseline Mean:  {response.mean_latency_baseline:.1f}")
            print(f"  Overhead:       {response.get_latency_overhead():+.1%}")
            print(f"  DPR-RC P95:     {response.p95_latency_dprrc:.1f}")
            print(f"  Baseline P95:   {response.p95_latency_baseline:.1f}")
            print()
            print(f"Report: {response.report_path}")
            print(f"Dataset: {response.dataset_path}")
            print("=" * 60)
            print()
            print("Benchmark completed successfully!")
            return 0
        else:
            print(f"ERROR: {response.error}")
            print("=" * 60)
            print()
            print("Benchmark failed!")
            return 1

    except Exception as e:
        print()
        print("=" * 60)
        print("BENCHMARK FAILED")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
