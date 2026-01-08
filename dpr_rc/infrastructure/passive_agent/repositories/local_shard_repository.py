"""
Infrastructure: Local Shard Repository

Loads shards from local dataset files and populates ChromaDB.
"""

import os
import json
from typing import Dict, List, Any, Optional

from dpr_rc.logging_utils import StructuredLogger, ComponentType
from .chromadb_repository import ChromaDBRepository


class LocalShardRepository:
    """
    Repository for loading shards from local dataset files.

    Supports lazy loading: shards are loaded on-demand when first accessed.
    """

    def __init__(
        self,
        embedding_repository: ChromaDBRepository,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize local shard repository.

        Args:
            embedding_repository: ChromaDB repository for vector storage
            embedding_model: Model for embeddings
        """
        self.embedding_repo = embedding_repository
        self.embedding_model = embedding_model
        self.logger = StructuredLogger(ComponentType.PASSIVE_WORKER)

        # Track loaded shards
        self._loaded_shards: Dict[str, Dict[str, Any]] = {}

    def load_shard(self, shard_id: str) -> bool:
        """
        Load a shard from local dataset into ChromaDB.

        Args:
            shard_id: Shard identifier (e.g., "shard_2015")

        Returns:
            True if shard was loaded successfully
        """
        if shard_id in self._loaded_shards:
            return True

        self.logger.logger.info(f"Loading shard on-demand: {shard_id}")

        # Check for local dataset
        local_path = os.getenv("LOCAL_DATASET_PATH")
        if not local_path or not os.path.exists(local_path):
            self.logger.logger.warning(f"LOCAL_DATASET_PATH not set or not found: {local_path}")
            return False

        self.logger.logger.info(f"Loading from LOCAL_DATASET_PATH: {local_path}")

        try:
            with open(local_path, 'r') as f:
                dataset = json.load(f)

            # Extract year from shard_id (e.g., "shard_2015" -> "2015")
            shard_year = shard_id.replace("shard_", "")

            # Get events for this shard year
            events = dataset.get("events", [])
            shard_events = [
                e for e in events
                if e.get("timestamp", "").startswith(shard_year)
            ]

            if not shard_events:
                self.logger.logger.warning(f"No events found for shard {shard_id}")
                return False

            # Convert events to documents
            documents = []
            for event in shard_events:
                doc = {
                    "id": event.get("id", ""),
                    "content": event.get("content", ""),
                    "metadata": {
                        "timestamp": event.get("timestamp", ""),
                        "source": "local_dataset",
                    }
                }
                documents.append(doc)

            # Insert into ChromaDB
            inserted = self.embedding_repo.bulk_insert(shard_id, documents)

            # Track loaded shard
            self._loaded_shards[shard_id] = {
                "document_count": inserted,
                "loaded_from": "local_dataset",
            }

            self.logger.logger.info(f"Loaded {inserted} documents for {shard_id}")
            return True

        except Exception as e:
            self.logger.logger.error(f"Failed to load shard {shard_id}: {e}")
            return False

    def is_shard_loaded(self, shard_id: str) -> bool:
        """Check if a shard is loaded."""
        return shard_id in self._loaded_shards

    def get_shard_data(self, shard_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a loaded shard."""
        return self._loaded_shards.get(shard_id)

    def query_shard(
        self, shard_id: str, query_text: str, n_results: int = 3
    ) -> List[Any]:
        """
        Query a shard for relevant documents.

        Args:
            shard_id: Shard to query
            query_text: Query text
            n_results: Number of results

        Returns:
            List of retrieval results
        """
        # Ensure shard is loaded
        if not self.is_shard_loaded(shard_id):
            self.load_shard(shard_id)

        return self.embedding_repo.query(shard_id, query_text, n_results)
