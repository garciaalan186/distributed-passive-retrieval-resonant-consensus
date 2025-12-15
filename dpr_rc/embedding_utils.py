"""
Embedding Utilities for DPR-RC System

Handles:
1. Computing embeddings using sentence-transformers
2. Storing/loading pre-computed embeddings in numpy format
3. Model versioning for future embedding model upgrades

GCS Storage Structure:
    gs://{bucket}/
    ├── raw/
    │   └── {scale}/
    │       └── shards/
    │           ├── shard_2020.json      # Plain text events
    │           └── shard_2021.json
    └── embeddings/
        └── {model_id}/
            └── {scale}/
                └── shards/
                    ├── shard_2020.npz   # Pre-computed vectors + doc IDs
                    └── shard_2021.npz
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import tempfile

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2


@dataclass
class EmbeddingMetadata:
    """Metadata for a set of embeddings"""
    model_id: str
    dimension: int
    num_documents: int
    shard_id: str
    created_at: str
    checksum: str


def get_model_folder_name(model_id: str) -> str:
    """Convert model ID to a valid folder name"""
    return model_id.replace("/", "_").replace(":", "_")


def compute_embeddings(
    texts: List[str],
    model_id: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model_id: Sentence-transformer model ID

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        embeddings = model.encode(texts, show_progress_bar=len(texts) > 100)
        return np.array(embeddings, dtype=np.float32)
    except ImportError as e:
        # Fallback: generate deterministic pseudo-embeddings for testing
        # These maintain semantic similarity properties through hash-based generation
        print(f"Warning: sentence-transformers not available (ImportError: {e}), using deterministic fallback")
        return _generate_fallback_embeddings(texts, EMBEDDING_DIMENSION)
    except Exception as e:
        # Catch other errors (e.g., model download failures)
        print(f"Warning: sentence-transformers failed ({type(e).__name__}: {e}), using deterministic fallback")
        return _generate_fallback_embeddings(texts, EMBEDDING_DIMENSION)


def _generate_fallback_embeddings(texts: List[str], dim: int) -> np.ndarray:
    """
    Generate deterministic pseudo-embeddings for testing.

    Uses hash-based generation that maintains some semantic properties:
    - Same text always produces same embedding
    - Similar texts produce somewhat similar embeddings (via word overlap)
    """
    embeddings = []

    for text in texts:
        # Create base vector from content hash
        content_hash = hashlib.sha256(text.encode()).digest()

        # Use hash bytes to seed random generator for reproducibility
        rng = np.random.RandomState(int.from_bytes(content_hash[:4], 'big'))
        base_vector = rng.randn(dim).astype(np.float32)

        # Add word-level components for basic semantic similarity
        words = set(text.lower().split())
        for word in list(words)[:20]:  # Limit to first 20 words
            word_hash = hashlib.md5(word.encode()).digest()
            word_seed = int.from_bytes(word_hash[:4], 'big')
            word_rng = np.random.RandomState(word_seed)
            base_vector += word_rng.randn(dim).astype(np.float32) * 0.1

        # Normalize to unit vector
        norm = np.linalg.norm(base_vector)
        if norm > 0:
            base_vector = base_vector / norm

        embeddings.append(base_vector)

    return np.array(embeddings, dtype=np.float32)


def save_embeddings_npz(
    embeddings: np.ndarray,
    doc_ids: List[str],
    texts: List[str],
    metadatas: List[Dict],
    output_path: str,
    model_id: str = DEFAULT_EMBEDDING_MODEL
) -> EmbeddingMetadata:
    """
    Save embeddings to numpy compressed format (.npz).

    Args:
        embeddings: numpy array of embeddings
        doc_ids: List of document IDs corresponding to each embedding
        texts: List of original texts (for verification)
        metadatas: List of metadata dicts for each document
        output_path: Path to save the .npz file
        model_id: Model ID used to generate embeddings

    Returns:
        EmbeddingMetadata object
    """
    from datetime import datetime

    # Compute checksum of embeddings for verification
    checksum = hashlib.md5(embeddings.tobytes()).hexdigest()[:12]

    # Create metadata
    metadata = EmbeddingMetadata(
        model_id=model_id,
        dimension=embeddings.shape[1],
        num_documents=len(doc_ids),
        shard_id=Path(output_path).stem,
        created_at=datetime.utcnow().isoformat(),
        checksum=checksum
    )

    # Save as compressed numpy archive
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        doc_ids=np.array(doc_ids, dtype=object),
        texts=np.array(texts, dtype=object),
        metadatas=np.array([json.dumps(m) for m in metadatas], dtype=object),
        metadata=json.dumps({
            "model_id": metadata.model_id,
            "dimension": metadata.dimension,
            "num_documents": metadata.num_documents,
            "shard_id": metadata.shard_id,
            "created_at": metadata.created_at,
            "checksum": metadata.checksum
        })
    )

    return metadata


def load_embeddings_npz(
    path: str
) -> Tuple[np.ndarray, List[str], List[str], List[Dict], EmbeddingMetadata]:
    """
    Load embeddings from numpy compressed format.

    Args:
        path: Path to .npz file

    Returns:
        Tuple of (embeddings, doc_ids, texts, metadatas, metadata)
    """
    data = np.load(path, allow_pickle=True)

    embeddings = data['embeddings']
    doc_ids = list(data['doc_ids'])
    texts = list(data['texts'])
    metadatas = [json.loads(m) for m in data['metadatas']]

    meta_dict = json.loads(str(data['metadata']))
    metadata = EmbeddingMetadata(
        model_id=meta_dict['model_id'],
        dimension=meta_dict['dimension'],
        num_documents=meta_dict['num_documents'],
        shard_id=meta_dict['shard_id'],
        created_at=meta_dict['created_at'],
        checksum=meta_dict['checksum']
    )

    return embeddings, doc_ids, texts, metadatas, metadata


class GCSEmbeddingStore:
    """
    Manages embeddings stored in Google Cloud Storage.

    Handles:
    - Uploading/downloading embeddings
    - Checking if embeddings exist for a model/shard
    - Lazy loading with local caching
    """

    def __init__(self, bucket_name: str, cache_dir: str = None):
        self.bucket_name = bucket_name
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="dpr_embeddings_")
        self._client = None
        self._bucket = None

    def _get_client(self):
        """Lazy-load GCS client"""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client()
                self._bucket = self._client.bucket(self.bucket_name)
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCS operations. "
                    "Install with: pip install google-cloud-storage"
                )
        return self._client, self._bucket

    def get_raw_path(self, scale: str, shard_id: str) -> str:
        """Get GCS path for raw JSON shard"""
        return f"raw/{scale}/shards/{shard_id}.json"

    def get_embeddings_path(self, model_id: str, scale: str, shard_id: str) -> str:
        """Get GCS path for embeddings"""
        model_folder = get_model_folder_name(model_id)
        return f"embeddings/{model_folder}/{scale}/shards/{shard_id}.npz"

    def embeddings_exist(self, model_id: str, scale: str, shard_id: str) -> bool:
        """Check if embeddings exist for a given model/scale/shard"""
        try:
            _, bucket = self._get_client()
            path = self.get_embeddings_path(model_id, scale, shard_id)
            blob = bucket.blob(path)
            return blob.exists()
        except Exception:
            return False

    def upload_raw_shard(self, local_path: str, scale: str, shard_id: str):
        """Upload raw JSON shard to GCS"""
        _, bucket = self._get_client()
        gcs_path = self.get_raw_path(scale, shard_id)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"  Uploaded raw: gs://{self.bucket_name}/{gcs_path}")

    def upload_embeddings(self, local_path: str, model_id: str, scale: str, shard_id: str):
        """Upload embeddings to GCS"""
        _, bucket = self._get_client()
        gcs_path = self.get_embeddings_path(model_id, scale, shard_id)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"  Uploaded embeddings: gs://{self.bucket_name}/{gcs_path}")

    def download_raw_shard(self, scale: str, shard_id: str) -> Optional[List[Dict]]:
        """Download raw JSON shard from GCS"""
        try:
            _, bucket = self._get_client()
            gcs_path = self.get_raw_path(scale, shard_id)
            blob = bucket.blob(gcs_path)

            if not blob.exists():
                return None

            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            print(f"Error downloading raw shard {shard_id}: {e}")
            return None

    def download_embeddings(
        self,
        model_id: str,
        scale: str,
        shard_id: str
    ) -> Optional[Tuple[np.ndarray, List[str], List[str], List[Dict], EmbeddingMetadata]]:
        """
        Download embeddings from GCS with local caching.

        Returns:
            Tuple of (embeddings, doc_ids, texts, metadatas, metadata) or None
        """
        try:
            _, bucket = self._get_client()
            gcs_path = self.get_embeddings_path(model_id, scale, shard_id)

            # Check local cache first
            cache_path = os.path.join(
                self.cache_dir,
                get_model_folder_name(model_id),
                scale,
                f"{shard_id}.npz"
            )

            if os.path.exists(cache_path):
                return load_embeddings_npz(cache_path)

            # Download from GCS
            blob = bucket.blob(gcs_path)
            if not blob.exists():
                return None

            # Ensure cache directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # Download to cache
            blob.download_to_filename(cache_path)

            return load_embeddings_npz(cache_path)
        except Exception as e:
            print(f"Error downloading embeddings for {shard_id}: {e}")
            return None

    def list_available_shards(self, scale: str) -> List[str]:
        """List available raw shards for a scale level"""
        try:
            _, bucket = self._get_client()
            prefix = f"raw/{scale}/shards/"
            blobs = bucket.list_blobs(prefix=prefix)

            shards = []
            for blob in blobs:
                if blob.name.endswith('.json'):
                    shard_id = Path(blob.name).stem
                    shards.append(shard_id)

            return sorted(shards)
        except Exception as e:
            print(f"Error listing shards: {e}")
            return []

    def list_embedding_models(self, scale: str) -> List[str]:
        """List embedding models that have embeddings for this scale"""
        try:
            _, bucket = self._get_client()
            prefix = f"embeddings/"
            blobs = bucket.list_blobs(prefix=prefix, delimiter='/')

            models = set()
            for page in blobs.pages:
                for prefix in page.prefixes:
                    # Extract model folder name
                    model_folder = prefix.split('/')[1]
                    # Check if it has embeddings for this scale
                    scale_prefix = f"embeddings/{model_folder}/{scale}/"
                    scale_blobs = list(bucket.list_blobs(prefix=scale_prefix, max_results=1))
                    if scale_blobs:
                        models.add(model_folder)

            return sorted(models)
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


def embed_query(query: str, model_id: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    Compute embedding for a single query.

    This should use the SAME model as was used to embed the documents.

    Args:
        query: Query string
        model_id: Model ID (must match document embeddings)

    Returns:
        1D numpy array of embedding
    """
    embeddings = compute_embeddings([query], model_id)
    return embeddings[0]
