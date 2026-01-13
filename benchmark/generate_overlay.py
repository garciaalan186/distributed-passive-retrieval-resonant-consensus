#!/usr/bin/env python3
"""
Generate Belief Revision Overlay for Existing Datasets

This script generates overlay events (belief revisions, contradictions)
for existing synthetic datasets. The overlay events are ingested into
the same ChromaDB shards as the original data.

Usage:
    # Generate overlay for a single scale
    python -m benchmark.generate_overlay --scale mini

    # Generate overlay for multiple scales
    python -m benchmark.generate_overlay --scale mini,small,medium

    # Generate and ingest into ChromaDB
    python -m benchmark.generate_overlay --scale mini --ingest

    # Custom revision rate
    python -m benchmark.generate_overlay --scale mini --revision-rate 0.10
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure imports work
sys.path.insert(0, os.getcwd())

from benchmark.synthetic.overlay_generator import OverlayGenerator


def generate_overlay(
    scale: str,
    revision_rate: float = 0.05,
    seed: int = 42,
    ingest: bool = False,
):
    """Generate overlay for a single scale."""
    base_dir = Path(f"benchmark_results_local/{scale}")
    dataset_path = base_dir / "dataset.json"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print(f"Run 'python -m benchmark.generate_dataset --scale {scale}' first.")
        return False

    print(f"\n{'='*60}")
    print(f"Generating Overlay: {scale.upper()}")
    print(f"{'='*60}")

    generator = OverlayGenerator(
        dataset_path=str(dataset_path),
        revision_rate=revision_rate,
        seed=seed,
    )

    overlay_data = generator.generate()
    generator.save(str(base_dir))

    if ingest:
        print("\nIngesting overlay events into ChromaDB...")
        ingest_overlay_events(scale, generator)

    return True


def ingest_overlay_events(scale: str, generator: OverlayGenerator):
    """Ingest overlay events into existing ChromaDB shards."""
    from dpr_rc.infrastructure.passive_agent.repositories import ChromaDBRepository

    # Set up ChromaDB path
    os.environ["CHROMA_DB_PATH"] = f"benchmark_results_local/chroma_db_{scale}"

    repo = ChromaDBRepository()
    events_for_ingestion = generator.get_events_for_ingestion()

    # Group by year/shard
    shards = {}
    for event in events_for_ingestion:
        ts = event.get("timestamp", "")
        if ts and len(ts) >= 4:
            year = ts[:4]
            shard_id = f"shard_{year}"
            if shard_id not in shards:
                shards[shard_id] = []
            shards[shard_id].append({
                "id": event["id"],
                "content": event["content"],
                "metadata": {
                    "id": event["id"],
                    "doc_id": event["id"],
                    "year": int(year),
                    "timestamp": event["timestamp"],
                    "domain": event.get("topic", "general"),
                    "type": "revision",
                    "is_overlay": True,
                }
            })

    total_inserted = 0
    for shard_id, docs in shards.items():
        try:
            count = repo.bulk_insert(shard_id, docs)
            total_inserted += count
            if count > 0:
                print(f"  {shard_id}: Inserted {count} revision events")
        except Exception as e:
            print(f"Error inserting to {shard_id}: {e}")

    print(f"Ingestion complete. Total revision events: {total_inserted}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate belief revision overlay for existing datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmark.generate_overlay --scale mini
    python -m benchmark.generate_overlay --scale mini,small --ingest
    python -m benchmark.generate_overlay --scale xxxl --revision-rate 0.03
        """
    )

    parser.add_argument(
        "--scale",
        type=str,
        required=True,
        help="Scale(s) to generate overlay for (comma-separated)"
    )

    parser.add_argument(
        "--revision-rate",
        type=float,
        default=0.05,
        help="Fraction of claims to revise per domain (default: 0.05)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Also ingest overlay events into ChromaDB"
    )

    args = parser.parse_args()

    scales = [s.strip() for s in args.scale.split(",")]

    success_count = 0
    for scale in scales:
        try:
            if generate_overlay(
                scale=scale,
                revision_rate=args.revision_rate,
                seed=args.seed,
                ingest=args.ingest,
            ):
                success_count += 1
        except Exception as e:
            print(f"Error generating overlay for {scale}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Overlay generation complete: {success_count}/{len(scales)} scales")
    print(f"{'='*60}")

    return 0 if success_count == len(scales) else 1


if __name__ == "__main__":
    sys.exit(main())
