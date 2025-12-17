#!/usr/bin/env python3
"""
Compare Old vs New Benchmark Implementations

This script runs benchmarks using both the old ResearchBenchmarkSuite
and the new RunBenchmarkUseCase to verify they produce equivalent results.

This is useful for:
1. Validating the new clean architecture implementation
2. Ensuring no regression in benchmark accuracy
3. Demonstrating backward compatibility
4. Identifying any behavioral differences

Usage:
    python -m benchmark.presentation.cli.compare_implementations --scale small

The script will:
1. Run benchmark using old ResearchBenchmarkSuite
2. Run benchmark using new RunBenchmarkUseCase
3. Compare the results
4. Report any significant differences
"""

import argparse
import asyncio
import sys
import os
import time
from pathlib import Path

# Old implementation
from benchmark.research_benchmark import ResearchBenchmarkSuite

# New implementation
from benchmark.application.dtos import RunBenchmarkRequest
from benchmark.infrastructure.factories import BenchmarkFactory


def compare_metrics(old_result: dict, new_result, metric_name: str, tolerance: float = 0.05):
    """
    Compare a metric between old and new implementations.

    Args:
        old_result: Result dictionary from old implementation
        new_result: RunBenchmarkResponse from new implementation
        metric_name: Name of metric to compare
        tolerance: Acceptable difference (e.g., 0.05 = 5%)

    Returns:
        Tuple of (matches, old_value, new_value, difference)
    """
    # Map metric names between old and new
    metric_mapping = {
        "dprrc_accuracy": ("dprrc_accuracy", "dprrc_accuracy"),
        "baseline_accuracy": ("baseline_accuracy", "baseline_accuracy"),
        "dprrc_hallucination_rate": ("dprrc_hallucination_rate", "dprrc_hallucination_rate"),
        "baseline_hallucination_rate": ("baseline_hallucination_rate", "baseline_hallucination_rate"),
        "dprrc_mean_latency": ("dprrc_mean_latency_ms", "mean_latency_dprrc"),
        "baseline_mean_latency": ("baseline_mean_latency_ms", "mean_latency_baseline"),
    }

    if metric_name not in metric_mapping:
        return False, None, None, None

    old_key, new_key = metric_mapping[metric_name]

    old_value = old_result.get(old_key, 0)
    new_value = getattr(new_result, new_key, 0)

    # Calculate difference
    if old_value == 0 and new_value == 0:
        difference = 0.0
        matches = True
    elif old_value == 0:
        difference = float('inf')
        matches = False
    else:
        difference = abs(new_value - old_value) / old_value
        matches = difference <= tolerance

    return matches, old_value, new_value, difference


def main():
    """Main comparison script"""
    parser = argparse.ArgumentParser(
        description="Compare old and new benchmark implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Benchmark scale level (default: small)"
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Acceptable difference between implementations (default: 0.10 = 10%%)"
    )

    parser.add_argument(
        "--skip-old",
        action="store_true",
        help="Skip running old implementation (use for testing new only)"
    )

    parser.add_argument(
        "--skip-new",
        action="store_true",
        help="Skip running new implementation (use for testing old only)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BENCHMARK IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print(f"Scale: {args.scale}")
    print(f"Tolerance: {args.tolerance:.1%}")
    print("=" * 80)
    print()

    old_result = None
    new_result = None

    # 1. Run old implementation
    if not args.skip_old:
        print("=" * 80)
        print("RUNNING OLD IMPLEMENTATION (ResearchBenchmarkSuite)")
        print("=" * 80)
        print()

        try:
            # Set environment for old implementation
            os.environ["BENCHMARK_SCALE"] = args.scale

            old_suite = ResearchBenchmarkSuite(output_dir="comparison_old")
            old_start = time.time()
            old_results_summary = old_suite.run_full_benchmark()
            old_elapsed = time.time() - old_start

            # Extract results for this scale
            old_result = None
            for scale_result in old_results_summary.get("scale_results", []):
                if scale_result["scale"] == args.scale:
                    old_result = scale_result
                    break

            if old_result:
                print()
                print("Old Implementation Results:")
                print(f"  DPR-RC Accuracy: {old_result['dprrc_accuracy']:.2%}")
                print(f"  Baseline Accuracy: {old_result['baseline_accuracy']:.2%}")
                print(f"  DPR-RC Hallucination Rate: {old_result['dprrc_hallucination_rate']:.2%}")
                print(f"  Baseline Hallucination Rate: {old_result['baseline_hallucination_rate']:.2%}")
                print(f"  DPR-RC Mean Latency: {old_result['dprrc_mean_latency_ms']:.1f} ms")
                print(f"  Baseline Mean Latency: {old_result['baseline_mean_latency_ms']:.1f} ms")
                print(f"  Elapsed Time: {old_elapsed:.2f}s")
                print()
            else:
                print("ERROR: Could not find results for scale '{args.scale}'")
                return 1

        except Exception as e:
            print(f"ERROR running old implementation: {e}")
            import traceback
            traceback.print_exc()
            if not args.skip_new:
                print("Continuing with new implementation...")
            else:
                return 1

    # 2. Run new implementation
    if not args.skip_new:
        print("=" * 80)
        print("RUNNING NEW IMPLEMENTATION (RunBenchmarkUseCase)")
        print("=" * 80)
        print()

        try:
            factory = BenchmarkFactory()
            use_case = factory.create_benchmark_use_case(
                scale=args.scale,
                output_dir="comparison_new",
                use_new_executor=False,  # Use HTTP mode for fair comparison
                baseline_mode="local",
                timeout=60.0
            )

            request = RunBenchmarkRequest(
                scale=args.scale,
                output_dir="comparison_new",
                enable_hallucination_detection=True,
                slm_service_url=os.getenv("SLM_SERVICE_URL", "http://localhost:8081"),
                seed=None  # Use random seed for fair comparison
            )

            new_start = time.time()
            new_result = asyncio.run(use_case.execute(request))
            new_elapsed = time.time() - new_start

            if new_result.succeeded:
                print()
                print("New Implementation Results:")
                print(f"  DPR-RC Accuracy: {new_result.dprrc_accuracy:.2%}")
                print(f"  Baseline Accuracy: {new_result.baseline_accuracy:.2%}")
                print(f"  DPR-RC Hallucination Rate: {new_result.dprrc_hallucination_rate:.2%}")
                print(f"  Baseline Hallucination Rate: {new_result.baseline_hallucination_rate:.2%}")
                print(f"  DPR-RC Mean Latency: {new_result.mean_latency_dprrc:.1f} ms")
                print(f"  Baseline Mean Latency: {new_result.mean_latency_baseline:.1f} ms")
                print(f"  Elapsed Time: {new_elapsed:.2f}s")
                print()
            else:
                print(f"ERROR: New implementation failed: {new_result.error}")
                return 1

        except Exception as e:
            print(f"ERROR running new implementation: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # 3. Compare results
    if old_result and new_result:
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print()

        metrics_to_compare = [
            "dprrc_accuracy",
            "baseline_accuracy",
            "dprrc_hallucination_rate",
            "baseline_hallucination_rate",
            "dprrc_mean_latency",
            "baseline_mean_latency",
        ]

        all_match = True
        comparison_table = []

        for metric in metrics_to_compare:
            matches, old_val, new_val, diff = compare_metrics(
                old_result, new_result, metric, args.tolerance
            )

            if matches:
                status = "✓ MATCH"
            else:
                status = "✗ DIFFER"
                all_match = False

            comparison_table.append({
                "metric": metric,
                "old": old_val,
                "new": new_val,
                "diff": diff,
                "status": status
            })

        # Print comparison table
        print(f"{'Metric':<30} {'Old':<12} {'New':<12} {'Diff':<10} {'Status':<10}")
        print("-" * 80)
        for row in comparison_table:
            metric_name = row["metric"].replace("_", " ").title()
            old_str = f"{row['old']:.4f}" if row['old'] is not None else "N/A"
            new_str = f"{row['new']:.4f}" if row['new'] is not None else "N/A"
            diff_str = f"{row['diff']:.2%}" if row['diff'] is not None and row['diff'] != float('inf') else "N/A"

            print(f"{metric_name:<30} {old_str:<12} {new_str:<12} {diff_str:<10} {row['status']:<10}")

        print()
        print("=" * 80)

        if all_match:
            print("✓ ALL METRICS MATCH within tolerance")
            print("The new implementation produces equivalent results!")
            print("=" * 80)
            return 0
        else:
            print("✗ SOME METRICS DIFFER beyond tolerance")
            print("Review the differences above. This may be expected if:")
            print("  - Different random seeds were used")
            print("  - Dataset generation has changed")
            print("  - Evaluation logic has been refined")
            print("=" * 80)
            return 1

    else:
        print("Skipped comparison (one or both implementations not run)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
