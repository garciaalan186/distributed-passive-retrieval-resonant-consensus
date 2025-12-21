"""
Infrastructure: GCS Shard Repository

Concrete implementation of IShardRepository using Google Cloud Storage.
Implements lazy loading with multiple fallback strategies.
"""

import os
import json
import threading
from typing import Dict, Optional, Any
import redis

from dpr_rc.domain.passive_agent.entities import ShardMetadata, LoadStrategy
from dpr_rc.domain.passive_agent.repositories import IEmbeddingRepository
from dpr_rc.embedding_utils import GCSEmbeddingStore, compute_embeddings, DEFAULT_EMBEDDING_MODEL
from dpr_rc.logging_utils import StructuredLogger, ComponentType


class GCSShardRepository:
    """
    Repository for loading shards from GCS with multiple fallback strategies.

    Lazy Loading Architecture:
    1. Check if shard already loaded (cache hit)
    2. Try pre-computed embeddings from GCS
    3. Try raw JSON from GCS (compute embeddings locally)
    4. Try Redis cache
    5. Generate fallback data

    Thread-safe with per-shard locks to prevent concurrent loading.
    """

    def __init__(
        self,
        embedding_repository: IEmbeddingRepository,
        bucket_name: Optional[str],
        scale: str = "medium",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        redis_client: Optional[redis.Redis] = None,
    ):
        """
        Initialize GCS shard repository.

        Args:
            embedding_repository: Repository for storing/querying embeddings
            bucket_name: GCS bucket name (None for local-only mode)
            scale: Data scale level (small/medium/large)
            embedding_model: Model for embeddings
            redis_client: Redis client for caching (optional)
        """
        self.embedding_repo = embedding_repository
        self.bucket_name = bucket_name
        self.scale = scale
        self.embedding_model = embedding_model
        self.redis_client = redis_client

        # GCS store (lazy initialized)
        self._gcs_store: Optional[GCSEmbeddingStore] = None

        # Track loaded shards
        self._loaded_shards: Dict[str, ShardMetadata] = {}
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        self.logger = StructuredLogger(ComponentType.PASSIVE_WORKER)

    def _get_gcs_store(self) -> Optional[GCSEmbeddingStore]:
        """Lazy initialize GCS store."""
        if not self.bucket_name:
            return None

        if not self._gcs_store:
            try:
                self._gcs_store = GCSEmbeddingStore(self.bucket_name)
            except Exception as e:
                self.logger.logger.warning(f"Failed to init GCS store: {e}")
                return None

        return self._gcs_store

    def _get_loading_lock(self, shard_id: str) -> threading.Lock:
        """Get or create a lock for a specific shard."""
        with self._global_lock:
            if shard_id not in self._loading_locks:
                self._loading_locks[shard_id] = threading.Lock()
            return self._loading_locks[shard_id]

    def load_shard(self, shard_id: str) -> ShardMetadata:
        """
        Load a shard using lazy loading with multiple fallback strategies.

        Args:
            shard_id: Identifier like "shard_2020"

        Returns:
            ShardMetadata describing the loaded shard

        Raises:
            Exception if all loading strategies fail
        """
        # Fast path: already loaded
        if shard_id in self._loaded_shards:
            return self._loaded_shards[shard_id]

        # Acquire per-shard lock
        lock = self._get_loading_lock(shard_id)
        with lock:
            # Double-check after acquiring lock
            if shard_id in self._loaded_shards:
                return self._loaded_shards[shard_id]

            self.logger.logger.info(f"Loading shard on-demand: {shard_id}")

            # Create metadata
            metadata = ShardMetadata.from_shard_id(shard_id)

            # Try loading strategies in order
            loaded = False
            gcs_store = self._get_gcs_store()

            # Strategy 1: Pre-computed embeddings from GCS
            if gcs_store and not loaded:
                loaded = self._load_from_gcs_embeddings(shard_id, metadata, gcs_store)

            # Strategy 2: Raw JSON from GCS
            if gcs_store and not loaded:
                loaded = self._load_from_gcs_raw(shard_id, metadata, gcs_store)

            # Strategy 3: Redis cache
            if self.redis_client and not loaded:
                loaded = self._load_from_redis_cache(shard_id, metadata)

            # Strategy 4: Fallback data
            if not loaded:
                loaded = self._generate_fallback_data(shard_id, metadata)

            if not loaded:
                raise Exception(f"Failed to load shard {shard_id} with all strategies")

            # Cache the metadata
            self._loaded_shards[shard_id] = metadata

            self.logger.logger.info(
                f"Shard {shard_id} loaded: {metadata.document_count} documents, "
                f"strategy={metadata.loaded_from}"
            )

            return metadata

    def _load_from_gcs_embeddings(
        self, shard_id: str, metadata: ShardMetadata, gcs_store: GCSEmbeddingStore
    ) -> bool:
        """Load pre-computed embeddings from GCS."""
        try:
            result = gcs_store.download_embeddings(
                model_id=self.embedding_model, scale=self.scale, shard_id=shard_id
            )

            if result is None:
                return False

            embeddings, doc_ids, texts, metadatas, embed_metadata = result

            self.logger.logger.info(
                f"Loaded pre-computed embeddings from GCS: "
                f"{embed_metadata.num_documents} docs, model={embed_metadata.model_id}"
            )

            # Insert into embedding repository
            count = self.embedding_repo.bulk_insert_with_embeddings(
                shard_id, doc_ids, texts, metadatas, embeddings
            )

            metadata.document_count = count
            metadata.embedding_model = embed_metadata.model_id
            metadata.loaded_from = LoadStrategy.GCS_EMBEDDINGS
            return True

        except Exception as e:
            self.logger.logger.warning(f"Failed to load embeddings from GCS: {e}")
            return False

    def _load_from_gcs_raw(
        self, shard_id: str, metadata: ShardMetadata, gcs_store: GCSEmbeddingStore
    ) -> bool:
        """Load raw JSON from GCS and compute embeddings locally."""
        try:
            events = gcs_store.download_raw_shard(scale=self.scale, shard_id=shard_id)

            if not events:
                return False

            self.logger.logger.info(
                f"Loaded raw JSON from GCS: {len(events)} events. "
                f"Computing embeddings locally..."
            )

            # Extract data
            texts = [event["content"] for event in events]
            doc_ids = [event["id"] for event in events]
            metadatas = [
                {
                    "timestamp": event.get("timestamp", ""),
                    "topic": event.get("topic", ""),
                    "event_type": event.get("event_type", ""),
                    "perspective": event.get("perspective", ""),
                }
                for event in events
            ]

            # Compute embeddings locally
            embeddings = compute_embeddings(texts, self.embedding_model)

            # Insert
            count = self.embedding_repo.bulk_insert_with_embeddings(
                shard_id, doc_ids, texts, metadatas, embeddings
            )

            metadata.document_count = count
            metadata.embedding_model = self.embedding_model
            metadata.loaded_from = LoadStrategy.GCS_RAW
            return True

        except Exception as e:
            self.logger.logger.warning(f"Failed to load raw from GCS: {e}")
            return False

    def _load_from_redis_cache(self, shard_id: str, metadata: ShardMetadata) -> bool:
        """Load from Redis cache."""
        if not self.redis_client:
            return False

        try:
            year = shard_id.replace("shard_", "")
            cache_key = f"dpr:history_cache:{year}"
            cached = self.redis_client.get(cache_key)

            if not cached:
                return False

            documents = json.loads(cached)
            self.logger.logger.info(f"Loaded {len(documents)} documents from Redis cache")

            count = self.embedding_repo.bulk_insert(shard_id, documents)

            metadata.document_count = count
            metadata.loaded_from = LoadStrategy.REDIS_CACHE
            return True

        except Exception as e:
            self.logger.logger.warning(f"Failed to load from Redis cache: {e}")
            return False

    def _generate_fallback_data(self, shard_id: str, metadata: ShardMetadata) -> bool:
        """Generate fallback data from local dataset or generic templates."""
        try:
            # Try loading from local benchmark dataset first
            dataset_path = f"benchmark_results_research/{self.scale}/dataset.json"

            if os.path.exists(dataset_path):
                if self._load_from_local_dataset(shard_id, metadata, dataset_path):
                    return True

            # Fall back to generic templates - REMOVED per user request
            # if a shard does not exist, it should fail.
            # return self._generate_generic_fallback(shard_id, metadata)
            
            raise Exception(f"Shard {shard_id} not found in GCS, Cache, or Local Dataset. Generic fallback disabled.")

        except Exception as e:
            self.logger.logger.warning(f"Failed to generate fallback data: {e}")
            return False

    def _load_from_local_dataset(
        self, shard_id: str, metadata: ShardMetadata, dataset_path: str
    ) -> bool:
        """Load real data from local benchmark dataset."""
        try:
            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            # Extract claims for this year
            year = metadata.year
            claims = dataset.get("claims", {})
            year_claims = [
                claim
                for claim in claims.values()
                if claim.get("timestamp", "").startswith(str(year))
            ]

            if not year_claims:
                return False

            # Prepare documents
            fallback_docs = []
            for i, claim in enumerate(year_claims):
                doc_id = f"fallback_{year}_{i}"
                content = claim.get("content", "")
                fallback_docs.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "metadata": {
                            "year": year,
                            "type": "fallback_dataset",
                            "claim_id": claim.get("id", ""),
                            "index": i,
                        },
                    }
                )

            count = self.embedding_repo.bulk_insert(shard_id, fallback_docs)

            metadata.document_count = count
            metadata.loaded_from = LoadStrategy.FALLBACK
            self.logger.logger.info(
                f"Generated {count} fallback docs from dataset (year={year})"
            )
            return True

        except Exception as e:
            self.logger.logger.warning(f"Failed to load from dataset: {e}")
            return False

    def _generate_generic_fallback(self, shard_id: str, metadata: ShardMetadata) -> bool:
        """Generate generic template fallback data."""
        year = metadata.year
        fallback_docs = []

        for i in range(10):
            doc_id = f"fallback_{year}_{i}"
            content = (
                f"Historical record from {year}: Research milestone {i} achieved. "
                f"Progress in domain area with metrics showing improvement. "
                f"Epoch {year} data point {i}."
            )
            fallback_docs.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": {"year": year, "type": "fallback_generic", "index": i},
                }
            )

        count = self.embedding_repo.bulk_insert(shard_id, fallback_docs)

        metadata.document_count = count
        metadata.loaded_from = LoadStrategy.FALLBACK
        self.logger.logger.warning(
            f"Generated {count} GENERIC fallback documents for year {year}. "
            f"Benchmark accuracy will be 0%."
        )
        return True

    def is_shard_loaded(self, shard_id: str) -> bool:
        """Check if a shard is already loaded."""
        return shard_id in self._loaded_shards

    def get_shard_data(self, shard_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a loaded shard."""
        if shard_id in self._loaded_shards:
            metadata = self._loaded_shards[shard_id]
            return {
                "shard_id": metadata.shard_id,
                "year": metadata.year,
                "document_count": metadata.document_count,
                "embedding_model": metadata.embedding_model,
                "loaded_from": metadata.loaded_from.value,
            }
        return None
