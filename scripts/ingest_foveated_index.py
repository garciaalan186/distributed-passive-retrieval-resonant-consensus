"""
Script: Ingest Foveated Index

Downloads foveated summaries (L3, L2, L1) from GCS and ingests them into
the local ChromaDB instance used by the Active Agent's FoveatedRouter.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from google.cloud import storage
import chromadb
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIRECTORY = "./foveated_index"
BUCKET_NAME = os.getenv("HISTORY_BUCKET", "dpr-history-data")
# Default embedding model matching benchmark/embedding_utils.py (implied)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

def download_foveated_files(bucket_name: str, scale: str, local_dir: Path):
    """Download foveated summary JSON files from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = f"foveated/{scale}/"
    blobs = bucket.list_blobs(prefix=prefix)

    local_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            filename = os.path.basename(blob.name)
            local_path = local_dir / filename
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)
            logger.info(f"Downloaded {filename}")
            
    return downloaded_files

def ingest_file(collection, file_path: Path, model):
    """Ingest a single foveated summary file into ChromaDB."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # data is Dict[str, Dict] -> summary_id: summary_obj
    ids = []
    documents = []
    metadatas = []

    for summary_id, content in data.items():
        # content is a FoveatedSummary dict: 
        # { "layer": "L3", "id": "...", "summary": "...", "time_range": [start, end], ... }
        
        text = content.get("summary", "")
        if not text:
            continue

        ids.append(summary_id)
        documents.append(text)
        
        # Prepare metadata for filtering
        meta = content.get("metadata", {}) or {}
        time_range = content.get("time_range")
        if time_range:
            meta["start_date"] = time_range[0]
            meta["end_date"] = time_range[1]
        
        # Ensure metadata values are primitives (Chroma requirement)
        clean_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        
        if content.get("layer") == "L2":
            # Add domain for filtering L2 by L3 results
            # The benchmark generator puts domain in metadata
            pass 
        if content.get("layer") == "L1":
             # Add epoch_id for filtering L1 by L2 results
             pass

        metadatas.append(clean_meta)

    if not ids:
        return

    # Compute embeddings matching benchmark model
    embeddings = model.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    logger.info(f"Ingested {len(ids)} items from {file_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Ingest foveated index from GCS")
    parser.add_argument("--scale", type=str, default="medium", help="Benchmark scale (small/medium/large)")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    args = parser.parse_args()

    # Check if index exists
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Collections
    col_l3 = chroma_client.get_or_create_collection("foveated_l3")
    col_l2 = chroma_client.get_or_create_collection("foveated_l2")
    col_l1 = chroma_client.get_or_create_collection("foveated_l1")

    if not args.force and col_l3.count() > 0:
        logger.info("Index already exists. Skipping ingestion. Use --force to overwrite.")
        return

    logger.info(f"Starting ingestion for scale={args.scale} from bucket={BUCKET_NAME}")

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Download files
    temp_dir = Path("./temp_foveated_data")
    files = download_foveated_files(BUCKET_NAME, args.scale, temp_dir)

    if not files:
        logger.warning("No files found! Check GCS bucket and path.")
        return

    # Ingest
    for file_path in files:
        if "L3" in file_path.name:
            ingest_file(col_l3, file_path, model)
        elif "L2" in file_path.name:
            ingest_file(col_l2, file_path, model)
        elif "L1" in file_path.name:
            ingest_file(col_l1, file_path, model)

    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()
