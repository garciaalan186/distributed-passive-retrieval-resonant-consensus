"""
Domain Service: Foveated Router

Implements semantic routing using hierarchical vector indices (L3 -> L2 -> L1).
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

# Type alias for TimeRange: (start_date, end_date) in YYYY-MM format
TimeRange = Tuple[str, str]


@dataclass
class FoveatedMatch:
    """Result of a foveated vector search."""
    summary_id: str
    layer: str
    score: float
    time_range: Optional[TimeRange]
    metadata: Dict


class IFoveatedVectorStore:
    """Interface for the foveated vector store adapter."""
    
    def search(
        self, 
        layer: str, 
        query_text: str, 
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[FoveatedMatch]:
        """Search a specific foveation layer."""
        ...


class FoveatedRouter:
    """
    Domain service for determining relevant time ranges using semantic search.
    
    Performs "Foveated Descent":
    1. Search L3 (Domain) -> Get candidate domains
    2. Search L2 (Epoch) -> Filter by candidate domains -> Get candidate epochs
    3. Search L1 (Shard) -> Filter by candidate epochs -> Get precise time ranges
    """

    def __init__(
        self, 
        vector_store: IFoveatedVectorStore,
        min_confidence: float = 0.7
    ):
        self.vector_store = vector_store
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)

    def get_semantic_time_ranges(self, query_text: str) -> Optional[List[TimeRange]]:
        """
        Get semantically relevant time ranges for a query.
        
        Returns None if no high-confidence matches are found (fallback to broadcast).
        """
        # 1. L3 Domain Search
        # Broad search to find relevant research domains
        l3_matches = self.vector_store.search("L3", query_text, limit=3)
        
        valid_l3 = [m for m in l3_matches if m.score >= self.min_confidence]
        
        if not valid_l3:
            self.logger.info("No high-confidence L3 matches found. Falling back to simple routing.")
            return None
            
        relevant_domains = [m.metadata.get("domain") for m in valid_l3 if m.metadata.get("domain")]
        
        # 2. L2 Epoch Search
        # Constrain search to identified domains
        l2_filter = {"domain": {"$in": relevant_domains}} if relevant_domains else None
        l2_matches = self.vector_store.search("L2", query_text, limit=5, filters=l2_filter)
        
        valid_l2 = [m for m in l2_matches if m.score >= self.min_confidence]
        
        if not valid_l2:
            # If we matched a domain but no specific epoch, return the domain's full ranges
            # But simpler to just return None and let simple routing handle it within the broad dates
            return None

        relevant_epochs = [m.metadata.get("epoch_id") for m in valid_l2 if m.metadata.get("epoch_id")]

        # 3. L1 Shard Summary Search
        # Constrain search to identified epochs
        l1_filter = {"epoch_id": {"$in": relevant_epochs}} if relevant_epochs else None
        l1_matches = self.vector_store.search("L1", query_text, limit=10, filters=l1_filter)
        
        valid_l1 = [m for m in l1_matches if m.score >= self.min_confidence]
        
        if not valid_l1:
            # Match at epoch level is good enough to return epoch ranges
            return [m.time_range for m in valid_l2 if m.time_range]
            
        # Return precise ranges from L1 matches
        return [m.time_range for m in valid_l1 if m.time_range]
