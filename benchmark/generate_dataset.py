#!/usr/bin/env python3
"""
Standalone dataset generation script for DPR-RC experiments.

Generates synthetic historical datasets at various scales without running
the full benchmark suite. Useful for pre-generating large datasets.

Usage:
    python3 -m benchmark.generate_dataset --scale xxxl
    python3 -m benchmark.generate_dataset --scale xl --output-dir /path/to/output
    python3 -m benchmark.generate_dataset --scale custom --events-per-year 1000 --domains 5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

from benchmark.synthetic.generator import SyntheticHistoryGeneratorV2
from benchmark.config import get_scale_config


def generate_dataset(
    scale: str = None,
    events_per_topic_per_year: int = None,
    num_domains: int = None,
    perspectives_per_event: int = 3,
    seed: int = 42,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Generate a synthetic dataset at the specified scale.

    Args:
        scale: Named scale (xs, s, m, l, xl, xxl, xxxl) or None for custom
        events_per_topic_per_year: Override events per topic per year
        num_domains: Override number of domains
        perspectives_per_event: Perspectives per event (default 3)
        seed: Random seed for reproducibility
        output_dir: Output directory (default: benchmark_results_local/{scale})

    Returns:
        Generated dataset dictionary
    """
    # Load scale config if specified
    if scale and scale != "custom":
        try:
            scale_config = get_scale_config(scale)
            if events_per_topic_per_year is None:
                events_per_topic_per_year = scale_config["events_per_topic_per_year"]
            if num_domains is None:
                num_domains = scale_config["num_domains"]
            if "perspectives_per_event" in scale_config:
                perspectives_per_event = scale_config["perspectives_per_event"]
        except KeyError:
            print(f"Warning: Scale '{scale}' not found in config, using provided values")

    # Validate required parameters
    if events_per_topic_per_year is None:
        events_per_topic_per_year = 100
    if num_domains is None:
        num_domains = 4

    # Calculate expected event count
    years = 11  # 2015-2025
    expected_events = events_per_topic_per_year * num_domains * years * perspectives_per_event

    print(f"Dataset Generation Configuration:")
    print(f"  Scale: {scale or 'custom'}")
    print(f"  Events per topic per year: {events_per_topic_per_year}")
    print(f"  Domains: {num_domains}")
    print(f"  Perspectives per event: {perspectives_per_event}")
    print(f"  Years: {years} (2015-2025)")
    print(f"  Expected events: ~{expected_events:,}")
    print(f"  Seed: {seed}")
    print()

    # Generate dataset
    print("Generating dataset...")
    start_time = time.time()

    generator = SyntheticHistoryGeneratorV2(
        events_per_topic_per_year=events_per_topic_per_year,
        num_domains=num_domains,
        perspectives_per_event=perspectives_per_event,
        seed=seed
    )

    dataset = generator.generate_dataset()

    generation_time = time.time() - start_time
    actual_events = len(dataset["events"])

    print(f"Generated {actual_events:,} events in {generation_time:.1f}s")
    print(f"  Queries: {len(dataset['queries'])}")
    print(f"  Claims: {len(dataset.get('claims', []))}")

    # Determine output directory
    if output_dir is None:
        output_dir = Path("benchmark_results_local") / (scale or "custom")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset
    print(f"\nSaving to {output_dir}/...")

    dataset_path = output_dir / "dataset.json"
    glossary_path = output_dir / "glossary.json"
    metadata_path = output_dir / "generation_metadata.json"

    # Save dataset
    print(f"  Writing dataset.json...")
    save_start = time.time()
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)
    dataset_size_mb = dataset_path.stat().st_size / (1024 * 1024)
    print(f"    Size: {dataset_size_mb:.1f} MB, Time: {time.time() - save_start:.1f}s")

    # Save glossary
    print(f"  Writing glossary.json...")
    with open(glossary_path, "w") as f:
        json.dump(generator.glossary, f, indent=2)

    # Save generation metadata
    metadata = {
        "scale": scale or "custom",
        "config": {
            "events_per_topic_per_year": events_per_topic_per_year,
            "num_domains": num_domains,
            "perspectives_per_event": perspectives_per_event,
            "seed": seed
        },
        "results": {
            "total_events": actual_events,
            "total_queries": len(dataset["queries"]),
            "generation_time_seconds": generation_time,
            "dataset_size_mb": dataset_size_mb
        },
        "output_files": {
            "dataset": str(dataset_path),
            "glossary": str(glossary_path)
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGeneration complete!")
    print(f"  Dataset: {dataset_path}")
    print(f"  Glossary: {glossary_path}")
    print(f"  Metadata: {metadata_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for DPR-RC experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate xxxl dataset (~1M events)
  python3 -m benchmark.generate_dataset --scale xxxl

  # Generate with custom parameters
  python3 -m benchmark.generate_dataset --scale custom --events-per-year 500 --domains 3

  # Generate to specific output directory
  python3 -m benchmark.generate_dataset --scale xl --output-dir ./my_datasets/xl

Available scales: xs, s, mini, small, m, medium, l, large, xl, xxl, xxxl
        """
    )

    parser.add_argument(
        "--scale", "-s",
        type=str,
        default="medium",
        help="Scale name (xs, s, m, l, xl, xxl, xxxl, or 'custom')"
    )

    parser.add_argument(
        "--events-per-year", "-e",
        type=int,
        default=None,
        help="Events per topic per year (overrides scale config)"
    )

    parser.add_argument(
        "--domains", "-d",
        type=int,
        default=None,
        help="Number of research domains (overrides scale config)"
    )

    parser.add_argument(
        "--perspectives", "-p",
        type=int,
        default=3,
        help="Perspectives per event (default: 3)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: benchmark_results_local/{scale})"
    )

    args = parser.parse_args()

    generate_dataset(
        scale=args.scale,
        events_per_topic_per_year=args.events_per_year,
        num_domains=args.domains,
        perspectives_per_event=args.perspectives,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
