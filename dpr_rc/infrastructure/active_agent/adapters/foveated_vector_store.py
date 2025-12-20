"""
Infrastructure Adapter: Foveated Vector Store

Implements the IFoveatedVectorStore interface using ChromaDB.
"""

from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from dpr_rc.domain.active_agent.services.foveated_router import IFoveatedVectorStore, FoveatedMatch

class ChromaFoveatedVectorStore(IFoveatedVectorStore):
    """
    ChromaDB implementation of the foveated vector store.
    """

    def __init__(self, persist_directory: str = "./foveated_index"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize collections for each layer
        self.collections = {
            "L3": self.client.get_or_create_collection("foveated_l3"),
            "L2": self.client.get_or_create_collection("foveated_l2"),
            "L1": self.client.get_or_create_collection("foveated_l1"),
        }

    def search(
        self, 
        layer: str, 
        query_text: str, 
        limit: int = 5,
        filters: Optional[Dict] = None
    ) -> List[FoveatedMatch]:
        """
        Search a specific foveation layer using ChromaDB.
        """
        if layer not in self.collections:
            raise ValueError(f"Invalid layer: {layer}")

        collection = self.collections[layer]
        
        # ChromaDB expects filters in a specific format
        # This is a simplified pass-through; complex filters might need translation
        chroma_filter = filters if filters else None

        results = collection.query(
            query_texts=[query_text],
            n_results=limit,
            where=chroma_filter
        )

        matches = []
        if not results['ids']:
            return matches

        # Parse results (unzipping the parallel lists returned by Chroma)
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]  # Chroma returns distances, not scores

        for i, id_ in enumerate(ids):
            meta = metadatas[i] or {}
            # Convert distance to similarity score (approximate)
            # Cosine distance: 0 is identical, 2 is opposite.
            # Score = 1 - (distance / 2)
            distance = distances[i]
            score = 1.0 - (distance / 2.0)

            # Extract time range from metadata
            start = meta.get("start_date")
            end = meta.get("end_date")
            time_range = (start, end) if start and end else None

            matches.append(FoveatedMatch(
                summary_id=id_,
                layer=layer,
                score=score,
                time_range=time_range,
                metadata=meta
            ))

        return matches
