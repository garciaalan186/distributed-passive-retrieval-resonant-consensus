"""
Domain Service: RFI Processor

Orchestrates Request For Information (RFI) processing.
Main domain logic: retrieve → verify → calculate → vote.
"""

import json
from typing import Dict, Any, List, Optional, Protocol
from ..entities import QuadrantCoordinates
from .verification_service import VerificationService
from .quadrant_service import QuadrantService


class RetrievalResult:
    """Result from document retrieval."""

    def __init__(self, content: str, distance: float, metadata: Dict[str, Any]):
        self.content = content
        self.distance = distance
        self.metadata = metadata


class IEmbeddingRepository(Protocol):
    """Interface for embedding repository operations."""

    def query(
        self, shard_id: str, query_text: str, n_results: int = 3
    ) -> List[RetrievalResult]:
        """Query for similar documents in a shard."""
        ...

    def collection_exists(self, shard_id: str) -> bool:
        """Check if collection exists."""
        ...


class Vote:
    """Domain entity representing a consensus vote."""

    def __init__(
        self,
        trace_id: str,
        worker_id: str,
        cluster_id: str,
        content_hash: str,
        confidence_score: float,
        binary_vote: int,
        semantic_quadrant: List[float],
        content_snippet: str,
        author_cluster: str,
    ):
        self.trace_id = trace_id
        self.worker_id = worker_id
        self.cluster_id = cluster_id
        self.content_hash = content_hash
        self.confidence_score = confidence_score
        self.binary_vote = binary_vote
        self.semantic_quadrant = semantic_quadrant
        self.content_snippet = content_snippet
        self.author_cluster = author_cluster

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "worker_id": self.worker_id,
            "cluster_id": self.cluster_id,
            "content_hash": self.content_hash,
            "confidence_score": self.confidence_score,
            "binary_vote": self.binary_vote,
            "semantic_quadrant": self.semantic_quadrant,
            "content_snippet": self.content_snippet,
            "author_cluster": self.author_cluster,
        }


class RFIProcessor:
    """
    Domain service for processing RFI (Request For Information).

    Orchestrates the main pipeline:
    1. Retrieve documents from embedding repository
    2. Verify using SLM
    3. Calculate semantic quadrant
    4. Create consensus vote
    """

    def __init__(
        self,
        embedding_repo: IEmbeddingRepository,
        verification_service: VerificationService,
        quadrant_service: QuadrantService,
        worker_id: str,
        cluster_id: str,
        vote_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize RFI processor.

        Args:
            embedding_repo: Repository for document retrieval
            verification_service: Service for L2 verification
            quadrant_service: Service for L3 quadrant calculation
            worker_id: Identifier for this worker
            cluster_id: Cluster this worker belongs to
            vote_threshold: Threshold for binary vote conversion
            confidence_threshold: Minimum confidence to cast vote
        """
        self.embedding_repo = embedding_repo
        self.verification_service = verification_service
        self.quadrant_service = quadrant_service
        self.worker_id = worker_id
        self.cluster_id = cluster_id
        self.vote_threshold = vote_threshold
        self.confidence_threshold = confidence_threshold

    def process_rfi(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        foveated_context: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Vote]:
        """
        Process RFI and generate votes.

        Query handling:
        - query_text: Enhanced query (from SLM) used for embedding-based retrieval
        - original_query: Original user query used for SLM verification

        Args:
            trace_id: Trace ID for logging correlation
            query_text: Enhanced query for retrieval
            original_query: Original query for verification
            target_shards: List of shard IDs to query
            foveated_context: Dict mapping shard_id → {'L1': summary, 'L2': epoch}

        Returns:
            List of Vote objects (can be empty if no relevant content)
        """
        votes = []

        # Process each target shard
        for shard_id in target_shards:
            # Retrieve from this shard using ENHANCED query
            results = self.embedding_repo.query(
                shard_id=shard_id, query_text=query_text, n_results=3
            )

            if not results or len(results) == 0:
                # No relevant documents found
                continue

            # Take top result
            top_result = results[0]

            # L2 Verification using ORIGINAL query
            # Extract depth from metadata
            depth = top_result.metadata.get("hierarchy_depth", 0)

            # Get foveated context for this shard
            shard_context = None
            if foveated_context and shard_id in foveated_context:
                shard_context = foveated_context[shard_id]

            verification_result = self.verification_service.verify(
                query=original_query,
                content=top_result.content,
                depth=depth,
                foveated_context=shard_context,
                trace_id=trace_id,
            )

            # Check confidence threshold
            adjusted_confidence = verification_result.adjusted_confidence
            if adjusted_confidence < self.confidence_threshold:
                # Below threshold, skip voting
                continue

            # RCP v4: Compute binary vote from confidence
            binary_vote = self.quadrant_service.compute_binary_vote(
                adjusted_confidence, self.vote_threshold
            )

            # Hash content for deduplication
            content_hash = self._hash_content(top_result.content)

            # RCP v4: Semantic quadrant (placeholder - active agent recalculates)
            # For now, use [0.0, 0.0] as active agent will compute from all votes
            quadrant = [0.0, 0.0]

            # Create vote
            vote = Vote(
                trace_id=trace_id,
                worker_id=self.worker_id,
                cluster_id=self.cluster_id,
                content_hash=content_hash,
                confidence_score=adjusted_confidence,
                binary_vote=binary_vote,
                semantic_quadrant=quadrant,
                content_snippet=top_result.content[:500],
                author_cluster=self.cluster_id,  # Worker is author of this artifact
            )

            votes.append(vote)

        return votes

    def _hash_content(self, content: str) -> str:
        """Hash content for deduplication (MD5 hex digest)."""
        import hashlib

        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
