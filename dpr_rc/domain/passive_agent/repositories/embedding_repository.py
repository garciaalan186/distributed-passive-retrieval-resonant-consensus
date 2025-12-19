"""
Repository Interface: Embedding Repository

Defines the contract for vector database operations (ChromaDB).
"""

from typing import Protocol, List, Dict, Any


class RetrievalResult:
    """Result from vector similarity search."""

    def __init__(self, content: str, distance: float, metadata: Dict[str, Any]):
        self.content = content
        self.distance = distance
        self.metadata = metadata


class IEmbeddingRepository(Protocol):
    """
    Interface for vector database operations.

    Handles document storage, retrieval, and similarity search
    in the embedding space.
    """

    def create_collection(self, shard_id: str) -> Any:
        """
        Create or get a vector collection for a shard.

        Args:
            shard_id: Identifier like "shard_2020"

        Returns:
            Collection object
        """
        ...

    def bulk_insert(
        self, shard_id: str, documents: List[Dict[str, Any]]
    ) -> int:
        """
        Insert multiple documents into a shard's collection.
        ChromaDB will compute embeddings automatically.

        Args:
            shard_id: Shard identifier
            documents: List of documents with 'id', 'content', 'metadata'

        Returns:
            Number of documents inserted
        """
        ...

    def bulk_insert_with_embeddings(
        self,
        shard_id: str,
        doc_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Any,  # numpy array
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
        ...

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
        ...

    def collection_exists(self, shard_id: str) -> bool:
        """
        Check if a collection exists for a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            True if collection exists
        """
        ...

    def get_collection_count(self, shard_id: str) -> int:
        """
        Get the number of documents in a shard's collection.

        Args:
            shard_id: Shard identifier

        Returns:
            Document count, or 0 if collection doesn't exist
        """
        ...
