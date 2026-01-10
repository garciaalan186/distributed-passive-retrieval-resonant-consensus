"""
Infrastructure: ChromaDB Embedding Repository

Concrete implementation of IEmbeddingRepository using ChromaDB for vector storage.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np

from dpr_rc.domain.passive_agent.repositories import IEmbeddingRepository, RetrievalResult
from dpr_rc.embedding_utils import embed_query, DEFAULT_EMBEDDING_MODEL
from dpr_rc.logging_utils import StructuredLogger, ComponentType
from dpr_rc.config import get_dpr_config

# Load query config for batch size
_query_config = get_dpr_config().get('query', {})


class ChromaDBRepository:
    """
    Repository for vector database operations using ChromaDB.

    Handles document storage, retrieval, and similarity search.
    Supports both automatic embedding computation and pre-computed embeddings.

    Uses CHROMA_DB_PATH environment variable for persistent storage if set,
    otherwise falls back to in-memory storage.
    """

    def __init__(
        self,
        chroma_client: Optional[chromadb.Client] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize ChromaDB repository.

        Args:
            chroma_client: ChromaDB client instance (creates new if None)
            embedding_model: Model to use for query embeddings

        If CHROMA_DB_PATH is set, uses persistent storage at that path.
        This allows multiple components (DPR-RC and baseline) to share data.
        """
        if chroma_client:
            self.chroma = chroma_client
        else:
            chroma_path = os.getenv("CHROMA_DB_PATH")
            if chroma_path:
                # Use persistent storage so DPR-RC and baseline share the same data
                self.chroma = chromadb.PersistentClient(path=chroma_path)
            else:
                # Fall back to in-memory for tests
                self.chroma = chromadb.Client()

        self.embedding_model = embedding_model
        self.logger = StructuredLogger(ComponentType.PASSIVE_WORKER)

        # Track collections
        self._collections: Dict[str, chromadb.Collection] = {}

    def create_collection(self, shard_id: str) -> chromadb.Collection:
        """
        Create or get a vector collection for a shard.

        Args:
            shard_id: Tempo-normalized identifier like "shard_000_2015-01_2021-12"

        Returns:
            ChromaDB collection object
        """
        if shard_id in self._collections:
            return self._collections[shard_id]

        collection = self.chroma.get_or_create_collection(
            name=f"history_{shard_id}", metadata={"hnsw:space": "cosine"}
        )

        self._collections[shard_id] = collection
        return collection

    def bulk_insert(self, shard_id: str, documents: List[Dict[str, Any]]) -> int:
        """
        Insert multiple documents into a shard's collection.
        ChromaDB will compute embeddings automatically.

        Args:
            shard_id: Shard identifier
            documents: List of documents with 'id', 'content', 'metadata'

        Returns:
            Number of documents inserted
        """
        if not documents:
            return 0

        collection = self.create_collection(shard_id)

        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            doc_id = doc.get("id") or hashlib.md5(doc["content"].encode()).hexdigest()[:12]
            ids.append(doc_id)
            contents.append(doc["content"])
            metadatas.append(doc.get("metadata", {}))

        # Insert in batches to avoid memory issues
        batch_size = _query_config.get('batch_size', 1000)
        inserted = 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_contents = contents[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            try:
                collection.add(
                    ids=batch_ids, documents=batch_contents, metadatas=batch_metadatas
                )
                inserted += len(batch_ids)
            except Exception as e:
                error_msg = str(e).lower()
                if "duplicate" in error_msg or "already" in error_msg:
                    # Skip duplicates silently
                    pass
                else:
                    self.logger.logger.error(f"Failed to insert batch: {e}")

        return inserted

    def bulk_insert_with_embeddings(
        self,
        shard_id: str,
        doc_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:
        """
        Insert multiple documents with pre-computed embeddings.

        Args:
            shard_id: Shard identifier
            doc_ids: Document IDs
            texts: Document texts
            metadatas: Document metadata dicts
            embeddings: Pre-computed embedding vectors (numpy array)

        Returns:
            Number of documents inserted
        """
        collection = self.create_collection(shard_id)

        batch_size = _query_config.get('batch_size', 1000)
        inserted = 0

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size].tolist()

            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    embeddings=batch_embeddings,
                )
                inserted += len(batch_ids)
            except Exception as e:
                # Handle duplicate IDs
                if "duplicate" in str(e).lower() or "already" in str(e).lower():
                    # Insert only non-duplicates
                    added = self._insert_non_duplicates(
                        collection, batch_ids, batch_texts, batch_metadatas, batch_embeddings
                    )
                    inserted += added
                else:
                    self.logger.logger.error(f"Failed to insert batch: {e}")

        return inserted

    def _insert_non_duplicates(
        self,
        collection: chromadb.Collection,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
        embeddings: List[List[float]],
    ) -> int:
        """Insert only non-duplicate documents."""
        try:
            existing = collection.get(ids=ids)
            existing_ids = set(existing.get("ids", []))

            new_ids = []
            new_texts = []
            new_metadatas = []
            new_embeddings = []

            for idx, doc_id in enumerate(ids):
                if doc_id not in existing_ids:
                    new_ids.append(doc_id)
                    new_texts.append(texts[idx])
                    new_metadatas.append(metadatas[idx])
                    new_embeddings.append(embeddings[idx])

            if new_ids:
                collection.add(
                    ids=new_ids,
                    documents=new_texts,
                    metadatas=new_metadatas,
                    embeddings=new_embeddings,
                )
                return len(new_ids)

            return 0

        except Exception as e:
            self.logger.logger.warning(f"Could not add non-duplicate docs: {e}")
            return 0

    def query(
        self, shard_id: str, query_text: str, n_results: int = 3
    ) -> List[RetrievalResult]:
        """
        Perform similarity search in a shard's collection.

        Args:
            shard_id: Shard to search
            query_text: Query string
            n_results: Number of results to return

        Returns:
            List of retrieval results ordered by similarity
        """
        if not self.collection_exists(shard_id):
            return []

        collection = self._collections[shard_id]

        if collection.count() == 0:
            return []

        try:
            # Embed query using the same model as documents
            query_embedding = embed_query(query_text, self.embedding_model)

            # Query using pre-computed embedding vector
            results = collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=n_results
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            # Convert to RetrievalResult objects
            retrieval_results = []
            docs = results["documents"][0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i, doc in enumerate(docs):
                distance = distances[i] if i < len(distances) else 0.0
                metadata = metadatas[i] if i < len(metadatas) else {}

                retrieval_results.append(RetrievalResult(content=doc, distance=distance, metadata=metadata))

            return retrieval_results

        except Exception as e:
            self.logger.logger.error(f"Query error: {e}")
            return []

    def collection_exists(self, shard_id: str) -> bool:
        """
        Check if a collection exists for a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            True if collection exists
        """
        # First check local cache
        if shard_id in self._collections:
            return True

        # Then check the actual ChromaDB (for persistent storage)
        collection_name = f"history_{shard_id}"
        try:
            existing_collections = self.chroma.list_collections()
            for col in existing_collections:
                if col.name == collection_name:
                    # Load it into cache for future use
                    self._collections[shard_id] = self.chroma.get_collection(collection_name)
                    return True
        except Exception:
            pass

        return False

    def get_collection_count(self, shard_id: str) -> int:
        """
        Get the number of documents in a shard's collection.

        Args:
            shard_id: Shard identifier

        Returns:
            Document count, or 0 if collection doesn't exist
        """
        if not self.collection_exists(shard_id):
            return 0

        try:
            return self._collections[shard_id].count()
        except Exception as e:
            self.logger.logger.warning(f"Failed to get count for {shard_id}: {e}")
            return 0
