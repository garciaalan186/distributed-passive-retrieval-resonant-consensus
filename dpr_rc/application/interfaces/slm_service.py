"""
ISLMService Interface

Interface for SLM (Small Language Model) query enhancement service.
This abstraction allows the use case to be independent of the implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class ISLMService(ABC):
    """
    Interface for SLM query enhancement service.

    The SLM service enhances queries by:
    - Expanding abbreviations (ML -> machine learning)
    - Adding synonyms for better recall
    - Clarifying ambiguous terms
    - Incorporating temporal context

    This interface enables dependency inversion: the use case depends
    on this abstraction, not on concrete HTTP client implementations.
    """

    @abstractmethod
    def enhance_query(
        self,
        query_text: str,
        timestamp_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance a query for better retrieval.

        Args:
            query_text: Original query text
            timestamp_context: Optional temporal context (e.g., "2023-01-01")

        Returns:
            Dict containing:
                - original_query: The input query
                - enhanced_query: The enhanced query text
                - expansions: List of expansions/synonyms added
                - enhancement_used: Whether enhancement was actually performed
                - inference_time_ms: Optional timing information

        Note:
            Implementations should be fault-tolerant and return the original
            query if enhancement fails.
        """
        pass
