"""
Generate synthetic history data and upload to GCS for reuse.

This script:
1. Generates phonotactic synthetic history data (once)
2. Uploads to GCS bucket for persistence
3. Workers download and ingest on startup (no regeneration)

Usage:
    python -m benchmark.generate_and_upload [--force]

    --force: Regenerate even if data already exists in GCS
"""

import os
import json
import argparse
from pathlib import Path
from google.cloud import storage

from .synthetic_history import SyntheticHistoryGeneratorV2

# Configuration
HISTORY_BUCKET = os.getenv("HISTORY_BUCKET", "dpr-history-data")
DATA_PREFIX = "synthetic_history/v2"
SCALE_CONFIGS = {
    "small": {"events_per_topic_per_year": 10, "num_domains": 2},
    "medium": {"events_per_topic_per_year": 25, "num_domains": 3},
    "large": {"events_per_topic_per_year": 50, "num_domains": 4},
    "stress": {"events_per_topic_per_year": 100, "num_domains": 5},
}


def get_gcs_client():
    """Get GCS client."""
    return storage.Client()


def data_exists_in_gcs(scale: str) -> bool:
    """Check if data for a scale level already exists in GCS."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(HISTORY_BUCKET)
        blob = bucket.blob(f"{DATA_PREFIX}/{scale}/dataset.json")
        return blob.exists()
    except Exception as e:
        print(f"Warning: Could not check GCS: {e}")
        return False


def upload_to_gcs(local_path: Path, gcs_path: str):
    """Upload a file to GCS."""
    client = get_gcs_client()
    bucket = client.bucket(HISTORY_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"  Uploaded: gs://{HISTORY_BUCKET}/{gcs_path}")


def download_from_gcs(gcs_path: str) -> dict:
    """Download JSON data from GCS."""
    client = get_gcs_client()
    bucket = client.bucket(HISTORY_BUCKET)
    blob = bucket.blob(gcs_path)
    content = blob.download_as_text()
    return json.loads(content)


def generate_and_upload_scale(scale: str, config: dict, force: bool = False):
    """Generate data for a scale level and upload to GCS."""
    print(f"\n{'='*60}")
    print(f"Processing scale: {scale}")
    print(f"{'='*60}")

    # Check if already exists
    if not force and data_exists_in_gcs(scale):
        print(f"  Data already exists in GCS. Use --force to regenerate.")
        return

    # Generate dataset
    print(f"  Generating dataset...")
    generator = SyntheticHistoryGeneratorV2(
        events_per_topic_per_year=config["events_per_topic_per_year"],
        perspectives_per_event=3,
        num_domains=config["num_domains"],
        seed=42  # Fixed seed for reproducibility
    )

    dataset = generator.generate_dataset()
    glossary = generator.glossary

    print(f"  Generated {len(dataset['events'])} events, {len(dataset['queries'])} queries")

    # Save locally first
    local_dir = Path(f"benchmark_data/{scale}")
    local_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = local_dir / "dataset.json"
    glossary_path = local_dir / "glossary.json"

    with open(dataset_path, "w") as f:
        json.dump(dataset, f)

    with open(glossary_path, "w") as f:
        json.dump(glossary, f)

    # Upload to GCS
    print(f"  Uploading to GCS...")
    upload_to_gcs(dataset_path, f"{DATA_PREFIX}/{scale}/dataset.json")
    upload_to_gcs(glossary_path, f"{DATA_PREFIX}/{scale}/glossary.json")

    # Also upload events as individual shard files for worker ingestion
    print(f"  Creating shard files...")
    events_by_year = {}
    for event in dataset['events']:
        year = event['timestamp'][:4]
        if year not in events_by_year:
            events_by_year[year] = []
        events_by_year[year].append(event)

    for year, events in events_by_year.items():
        shard_path = local_dir / f"shard_{year}.json"
        with open(shard_path, "w") as f:
            json.dump(events, f)
        upload_to_gcs(shard_path, f"{DATA_PREFIX}/{scale}/shards/shard_{year}.json")

    print(f"  Done! {len(events_by_year)} shards created.")


def list_available_data():
    """List all available data in GCS."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(HISTORY_BUCKET)

        print(f"\nAvailable data in gs://{HISTORY_BUCKET}/{DATA_PREFIX}/:")
        print("-" * 50)

        for scale in SCALE_CONFIGS.keys():
            blob = bucket.blob(f"{DATA_PREFIX}/{scale}/dataset.json")
            if blob.exists():
                blob.reload()
                size_mb = blob.size / (1024 * 1024)
                print(f"  {scale}: {size_mb:.2f} MB (last modified: {blob.updated})")
            else:
                print(f"  {scale}: Not generated")

    except Exception as e:
        print(f"Error listing GCS data: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate and upload synthetic history data")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if data exists")
    parser.add_argument("--scale", type=str, choices=list(SCALE_CONFIGS.keys()) + ["all"],
                        default="all", help="Scale level to generate")
    parser.add_argument("--list", action="store_true", help="List available data in GCS")

    args = parser.parse_args()

    if args.list:
        list_available_data()
        return

    print("=" * 60)
    print("SYNTHETIC HISTORY DATA GENERATOR")
    print(f"Target bucket: gs://{HISTORY_BUCKET}/{DATA_PREFIX}/")
    print("=" * 60)

    if args.scale == "all":
        for scale, config in SCALE_CONFIGS.items():
            generate_and_upload_scale(scale, config, args.force)
    else:
        generate_and_upload_scale(args.scale, SCALE_CONFIGS[args.scale], args.force)

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    list_available_data()


if __name__ == "__main__":
    main()
