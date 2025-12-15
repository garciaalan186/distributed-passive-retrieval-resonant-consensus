"""
Generate synthetic history data and upload to GCS for reuse.

This script:
1. Generates phonotactic synthetic history data (once)
2. Saves raw JSON (text only) to GCS
3. Computes embeddings and saves to separate model-specific folder
4. Workers download and ingest on startup (no regeneration)

GCS Storage Structure:
    gs://{bucket}/
    ├── raw/
    │   └── {scale}/
    │       ├── dataset.json         # Full dataset metadata
    │       ├── glossary.json        # Term glossary
    │       └── shards/
    │           ├── shard_2020.json  # Plain text events by year
    │           └── shard_2021.json
    └── embeddings/
        └── {model_id}/
            └── {scale}/
                └── shards/
                    ├── shard_2020.npz  # Pre-computed vectors
                    └── shard_2021.npz

Usage:
    python -m benchmark.generate_and_upload [--force] [--scale medium] [--skip-embeddings]

    --force: Regenerate even if data already exists in GCS
    --scale: Scale level (small, medium, large, stress, all)
    --skip-embeddings: Only upload raw JSON, skip embedding computation
"""

# Python 3.9 compatibility: patch importlib.metadata for chromadb
import importlib.metadata
if not hasattr(importlib.metadata, 'packages_distributions'):
    # Provide a fallback for Python < 3.10
    def _packages_distributions():
        return {}
    importlib.metadata.packages_distributions = _packages_distributions

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set before any HuggingFace imports to prevent fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for google-cloud-storage before importing
try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    print("Warning: google-cloud-storage not installed. Install with: pip install google-cloud-storage")

from .synthetic_history import SyntheticHistoryGeneratorV2

# Import embedding utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from dpr_rc.embedding_utils import (
    compute_embeddings,
    save_embeddings_npz,
    DEFAULT_EMBEDDING_MODEL,
    get_model_folder_name
)

# Configuration
HISTORY_BUCKET = os.getenv("HISTORY_BUCKET", "dpr-history-data")

SCALE_CONFIGS = {
    "small": {"events_per_topic_per_year": 10, "num_domains": 2},
    "medium": {"events_per_topic_per_year": 25, "num_domains": 3},
    "large": {"events_per_topic_per_year": 50, "num_domains": 4},
    "stress": {"events_per_topic_per_year": 100, "num_domains": 5},
}


def get_gcs_client():
    """Get GCS client."""
    if not HAS_GCS:
        raise ImportError(
            "google-cloud-storage is required. Install with: pip install google-cloud-storage"
        )
    return storage.Client()


def raw_data_exists(scale: str) -> bool:
    """Check if raw data for a scale level already exists in GCS."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(HISTORY_BUCKET)
        blob = bucket.blob(f"raw/{scale}/dataset.json")
        return blob.exists()
    except Exception as e:
        print(f"Warning: Could not check GCS: {e}")
        return False


def embeddings_exist(scale: str, model_id: str = DEFAULT_EMBEDDING_MODEL) -> bool:
    """Check if embeddings exist for a scale/model combination."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(HISTORY_BUCKET)
        model_folder = get_model_folder_name(model_id)
        # Check for at least one shard
        blob = bucket.blob(f"embeddings/{model_folder}/{scale}/shards/shard_2020.npz")
        return blob.exists()
    except Exception as e:
        print(f"Warning: Could not check embeddings in GCS: {e}")
        return False


def upload_to_gcs(local_path: Path, gcs_path: str):
    """Upload a file to GCS."""
    client = get_gcs_client()
    bucket = client.bucket(HISTORY_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"  Uploaded: gs://{HISTORY_BUCKET}/{gcs_path}")


def generate_raw_data(scale: str, config: dict, local_dir: Path) -> Dict:
    """Generate raw synthetic data and save locally."""
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

    # Save full dataset
    dataset_path = local_dir / "dataset.json"
    glossary_path = local_dir / "glossary.json"

    with open(dataset_path, "w") as f:
        json.dump(dataset, f)

    with open(glossary_path, "w") as f:
        json.dump(glossary, f)

    # Create shard files (events grouped by year)
    shards_dir = local_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    events_by_year = {}
    for event in dataset['events']:
        year = event['timestamp'][:4]
        if year not in events_by_year:
            events_by_year[year] = []
        events_by_year[year].append(event)

    for year, events in events_by_year.items():
        shard_path = shards_dir / f"shard_{year}.json"
        with open(shard_path, "w") as f:
            json.dump(events, f)

    print(f"  Created {len(events_by_year)} shard files")

    return {
        "dataset_path": dataset_path,
        "glossary_path": glossary_path,
        "shards_dir": shards_dir,
        "events_by_year": events_by_year
    }


def upload_raw_to_gcs(scale: str, local_data: Dict):
    """Upload raw JSON data to GCS."""
    print(f"  Uploading raw data to GCS...")

    # Upload dataset and glossary
    upload_to_gcs(local_data["dataset_path"], f"raw/{scale}/dataset.json")
    upload_to_gcs(local_data["glossary_path"], f"raw/{scale}/glossary.json")

    # Upload shard files
    for shard_file in local_data["shards_dir"].glob("*.json"):
        upload_to_gcs(shard_file, f"raw/{scale}/shards/{shard_file.name}")


def _embed_single_shard(args: Tuple[str, List[Dict], str, str, str]) -> Tuple[str, bool, str]:
    """
    Embed a single shard - designed to be called in parallel.

    Each shard is completely independent - no cross-shard context needed.
    This enables horizontal scaling across time shards.

    Args:
        args: Tuple of (year, events, scale, model_id, local_dir_str)

    Returns:
        Tuple of (year, success, message)
    """
    year, events, scale, model_id, local_dir_str = args
    local_dir = Path(local_dir_str)

    try:
        # Import inside function for multiprocessing compatibility
        from dpr_rc.embedding_utils import (
            compute_embeddings as _compute_embeddings,
            save_embeddings_npz as _save_embeddings_npz,
            get_model_folder_name as _get_model_folder_name
        )

        model_folder = _get_model_folder_name(model_id)
        embeddings_dir = local_dir / "embeddings" / model_folder
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Extract texts and metadata
        texts = [event['content'] for event in events]
        doc_ids = [event['id'] for event in events]
        metadatas = [
            {
                "timestamp": event.get('timestamp', ''),
                "topic": event.get('topic', ''),
                "event_type": event.get('event_type', ''),
                "perspective": event.get('perspective', '')
            }
            for event in events
        ]

        # Compute embeddings (completely independent of other shards)
        embeddings = _compute_embeddings(texts, model_id)

        # Save locally
        local_npz = embeddings_dir / f"shard_{year}.npz"
        _save_embeddings_npz(
            embeddings=embeddings,
            doc_ids=doc_ids,
            texts=texts,
            metadatas=metadatas,
            output_path=str(local_npz),
            model_id=model_id
        )

        # Upload to GCS
        gcs_path = f"embeddings/{model_folder}/{scale}/shards/shard_{year}.npz"

        # Get GCS client inside worker
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket(HISTORY_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_npz))

        return (year, True, f"shard_{year}: {len(events)} events embedded")

    except Exception as e:
        return (year, False, f"shard_{year}: Error - {str(e)}")


def compute_and_upload_embeddings(
    scale: str,
    events_by_year: Dict[str, List[Dict]],
    local_dir: Path,
    model_id: str = DEFAULT_EMBEDDING_MODEL,
    parallel: bool = True,
    max_workers: int = None
):
    """
    Compute embeddings for all shards and upload to GCS.

    Supports parallel processing across time shards since each shard
    is completely independent - embeddings are computed per-document
    with no cross-shard context needed.

    Args:
        scale: Scale level (small, medium, large, stress)
        events_by_year: Dict mapping year to list of events
        local_dir: Local directory for temporary files
        model_id: Embedding model to use
        parallel: If True, process shards in parallel
        max_workers: Max parallel workers (default: CPU count)
    """
    print(f"  Computing embeddings with model: {model_id}")
    print(f"  Parallel processing: {parallel}")

    model_folder = get_model_folder_name(model_id)
    embeddings_dir = local_dir / "embeddings" / model_folder
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if parallel and len(events_by_year) > 1:
        # Parallel processing - each shard is independent
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(events_by_year))

        print(f"  Using {max_workers} parallel workers for {len(events_by_year)} shards")

        # Prepare arguments for parallel execution
        shard_args = [
            (year, events, scale, model_id, str(local_dir))
            for year, events in events_by_year.items()
        ]

        # Process shards in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_embed_single_shard, args): args[0]
                      for args in shard_args}

            for future in as_completed(futures):
                year = futures[future]
                try:
                    year_result, success, message = future.result()
                    status = "✓" if success else "✗"
                    print(f"    {status} {message}")
                except Exception as e:
                    print(f"    ✗ shard_{year}: Exception - {str(e)}")
    else:
        # Sequential processing (single shard or parallel disabled)
        for year, events in events_by_year.items():
            print(f"    Processing shard_{year} ({len(events)} events)...")

            # Extract texts and metadata
            texts = [event['content'] for event in events]
            doc_ids = [event['id'] for event in events]
            metadatas = [
                {
                    "timestamp": event.get('timestamp', ''),
                    "topic": event.get('topic', ''),
                    "event_type": event.get('event_type', ''),
                    "perspective": event.get('perspective', '')
                }
                for event in events
            ]

            # Compute embeddings
            embeddings = compute_embeddings(texts, model_id)

            # Save locally
            local_npz = embeddings_dir / f"shard_{year}.npz"
            save_embeddings_npz(
                embeddings=embeddings,
                doc_ids=doc_ids,
                texts=texts,
                metadatas=metadatas,
                output_path=str(local_npz),
                model_id=model_id
            )

            # Upload to GCS
            gcs_path = f"embeddings/{model_folder}/{scale}/shards/shard_{year}.npz"
            upload_to_gcs(local_npz, gcs_path)

    print(f"  Embeddings complete for {len(events_by_year)} shards")


def generate_and_upload_scale(
    scale: str,
    config: dict,
    force: bool = False,
    skip_embeddings: bool = False,
    model_id: str = DEFAULT_EMBEDDING_MODEL
):
    """Generate data for a scale level and upload to GCS."""
    print(f"\n{'='*60}")
    print(f"Processing scale: {scale}")
    print(f"{'='*60}")

    # Check if raw data already exists
    if not force and raw_data_exists(scale):
        print(f"  Raw data already exists in GCS. Use --force to regenerate.")

        # Check if embeddings need to be computed
        if not skip_embeddings and not embeddings_exist(scale, model_id):
            print(f"  But embeddings for {model_id} are missing. Computing...")
            # Download raw data and compute embeddings
            # This is a TODO - for now require force regeneration
            print(f"  TODO: Download raw and compute embeddings. Use --force for now.")
        return

    # Create local directory
    local_dir = Path(f"benchmark_data/{scale}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate raw data
    local_data = generate_raw_data(scale, config, local_dir)

    # Step 2: Upload raw data to GCS
    upload_raw_to_gcs(scale, local_data)

    # Step 3: Compute and upload embeddings
    if not skip_embeddings:
        compute_and_upload_embeddings(
            scale=scale,
            events_by_year=local_data["events_by_year"],
            local_dir=local_dir,
            model_id=model_id
        )

    print(f"  Done! Scale '{scale}' complete.")


def list_available_data():
    """List all available data in GCS (gracefully handles permission errors)."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(HISTORY_BUCKET)

        print(f"\n{'='*60}")
        print(f"Available data in gs://{HISTORY_BUCKET}/")
        print(f"{'='*60}")

        # List raw data
        print("\nRaw Data:")
        print("-" * 40)
        for scale in SCALE_CONFIGS.keys():
            try:
                blob = bucket.blob(f"raw/{scale}/dataset.json")
                if blob.exists():
                    blob.reload()
                    size_mb = blob.size / (1024 * 1024)
                    print(f"  {scale}: {size_mb:.2f} MB")

                    # Count shards
                    shard_blobs = list(bucket.list_blobs(prefix=f"raw/{scale}/shards/"))
                    print(f"    Shards: {len(shard_blobs)}")
                else:
                    print(f"  {scale}: Not generated")
            except Exception as e:
                print(f"  {scale}: Unable to check (permission issue)")

        # List embeddings
        print("\nEmbeddings:")
        print("-" * 40)

        try:
            # Find all model folders
            embedding_prefix = "embeddings/"
            seen_models = set()

            for blob in bucket.list_blobs(prefix=embedding_prefix):
                parts = blob.name.split('/')
                if len(parts) >= 2:
                    seen_models.add(parts[1])

            if not seen_models:
                print("  (none computed yet)")
            else:
                for model_folder in sorted(seen_models):
                    print(f"  Model: {model_folder}")
                    for scale in SCALE_CONFIGS.keys():
                        try:
                            shard_blobs = list(bucket.list_blobs(
                                prefix=f"embeddings/{model_folder}/{scale}/shards/"
                            ))
                            if shard_blobs:
                                total_size = sum(b.size for b in shard_blobs) / (1024 * 1024)
                                print(f"    {scale}: {len(shard_blobs)} shards, {total_size:.2f} MB")
                        except Exception:
                            print(f"    {scale}: Unable to check")
        except Exception as e:
            print(f"  Unable to list embeddings (permission issue)")

    except Exception as e:
        print(f"Note: Could not list GCS data (run may still have succeeded): {e}")


def retroactive_embed(scale: str, model_id: str, parallel: bool = True, max_workers: int = None):
    """
    Retroactively compute embeddings for existing raw data with a new model.

    This is the key feature for model evolution - when you want to try
    a new embedding model, run this to create embeddings without regenerating raw data.

    Each shard is processed independently and can be parallelized across multiple
    workers since embedding computation requires no cross-shard context.

    Args:
        scale: Scale level (small, medium, large, stress)
        model_id: Embedding model to use
        parallel: Enable parallel processing across shards
        max_workers: Maximum parallel workers (default: CPU count)
    """
    print(f"\n{'='*60}")
    print(f"Retroactive Embedding: {scale} with {model_id}")
    print(f"Parallel: {parallel}, Max workers: {max_workers or 'auto'}")
    print(f"{'='*60}")

    try:
        client = get_gcs_client()
        bucket = client.bucket(HISTORY_BUCKET)

        # Check if raw data exists
        if not raw_data_exists(scale):
            print(f"  Error: No raw data found for scale '{scale}'")
            return False

        # Check if embeddings already exist
        if embeddings_exist(scale, model_id):
            print(f"  Embeddings already exist for {model_id}. Skipping.")
            return True

        # Download raw shards
        print(f"  Downloading raw shards...")
        local_dir = Path(f"benchmark_data/{scale}")
        local_dir.mkdir(parents=True, exist_ok=True)

        events_by_year = {}
        shard_prefix = f"raw/{scale}/shards/"

        for blob in bucket.list_blobs(prefix=shard_prefix):
            if blob.name.endswith('.json'):
                year = Path(blob.name).stem.replace('shard_', '')
                content = blob.download_as_text()
                events_by_year[year] = json.loads(content)
                print(f"    Downloaded shard_{year}: {len(events_by_year[year])} events")

        # Compute and upload embeddings (parallelized across shards)
        compute_and_upload_embeddings(
            scale=scale,
            events_by_year=events_by_year,
            local_dir=local_dir,
            model_id=model_id,
            parallel=parallel,
            max_workers=max_workers
        )

        print(f"  Done! Retroactive embedding complete.")
        return True

    except Exception as e:
        print(f"Error during retroactive embedding: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_embeddings_exist(
    scale: str,
    model_id: str = DEFAULT_EMBEDDING_MODEL,
    parallel: bool = True,
    max_workers: int = None
) -> bool:
    """
    Ensure embeddings exist for the given scale and model.

    If raw data exists but embeddings don't, computes them.
    This is called by run_cloud_benchmark.sh before running tests.

    Returns:
        True if embeddings exist (or were successfully computed), False otherwise
    """
    print(f"\n--- Checking embeddings for {scale} with {model_id} ---")

    if embeddings_exist(scale, model_id):
        print(f"  ✓ Embeddings already exist")
        return True

    if not raw_data_exists(scale):
        print(f"  ✗ No raw data found. Run with --force to generate.")
        return False

    print(f"  → Raw data exists but embeddings missing. Computing...")
    return retroactive_embed(scale, model_id, parallel, max_workers)


def main():
    parser = argparse.ArgumentParser(description="Generate and upload synthetic history data")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if data exists")
    parser.add_argument("--scale", type=str,
                        choices=list(SCALE_CONFIGS.keys()) + ["all"],
                        default="all",
                        help="Scale level to generate")
    parser.add_argument("--list", action="store_true",
                        help="List available data in GCS")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding computation (raw JSON only)")
    parser.add_argument("--model", type=str,
                        default=DEFAULT_EMBEDDING_MODEL,
                        help=f"Embedding model ID (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--retroactive-embed", action="store_true",
                        help="Compute embeddings for existing raw data with a new model")
    parser.add_argument("--ensure-embeddings", action="store_true",
                        help="Ensure embeddings exist (compute if missing, skip if present)")
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="Enable parallel processing across time shards (default: True)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing (sequential mode)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: CPU count)")

    args = parser.parse_args()

    if not HAS_GCS:
        print("\nError: google-cloud-storage is required.")
        print("Install with: pip install google-cloud-storage")
        sys.exit(1)

    # Determine parallel mode
    parallel = not args.no_parallel

    if args.list:
        list_available_data()
        return

    if args.ensure_embeddings:
        # Called by run_cloud_benchmark.sh to ensure embeddings exist
        if args.scale == "all":
            success = all(
                ensure_embeddings_exist(scale, args.model, parallel, args.workers)
                for scale in SCALE_CONFIGS.keys()
            )
        else:
            success = ensure_embeddings_exist(args.scale, args.model, parallel, args.workers)

        sys.exit(0 if success else 1)

    if args.retroactive_embed:
        if args.scale == "all":
            for scale in SCALE_CONFIGS.keys():
                retroactive_embed(scale, args.model, parallel, args.workers)
        else:
            retroactive_embed(args.scale, args.model, parallel, args.workers)
        return

    print("=" * 60)
    print("SYNTHETIC HISTORY DATA GENERATOR v2")
    print(f"Target bucket: gs://{HISTORY_BUCKET}/")
    print(f"Embedding model: {args.model}")
    print(f"Parallel processing: {parallel}")
    print("=" * 60)

    if args.scale == "all":
        for scale, config in SCALE_CONFIGS.items():
            generate_and_upload_scale(
                scale, config,
                force=args.force,
                skip_embeddings=args.skip_embeddings,
                model_id=args.model
            )
    else:
        generate_and_upload_scale(
            args.scale,
            SCALE_CONFIGS[args.scale],
            force=args.force,
            skip_embeddings=args.skip_embeddings,
            model_id=args.model
        )

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    list_available_data()


if __name__ == "__main__":
    main()
