"""
IWorkerService Interface

Interface for communicating with passive workers to gather consensus votes.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dpr_rc.models import ConsensusVote


class IWorkerService(ABC):
    """
    Interface for passive worker communication.

    This abstraction allows the use case to be independent of the
    communication mechanism (HTTP, Redis, gRPC, etc.).

    The worker service is responsible for:
    1. Broadcasting RFIs (Request For Information) to passive workers
    2. Collecting consensus votes from workers
    3. Handling timeouts and failures gracefully
    """

    @abstractmethod
    async def gather_votes(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str] = None
    ) -> List[ConsensusVote]:
        """
        Gather consensus votes from passive workers.

        This method encapsulates the entire worker communication flow:
        - Broadcasting RFI to workers (via HTTP, Redis, or other mechanism)
        - Waiting for and collecting votes
        - Handling timeouts and errors

        Args:
            trace_id: Trace ID for logging
            query_text: Enhanced query for retrieval
            original_query: Original query for verification
            target_shards: List of shard IDs to query
            timestamp_context: Optional temporal context

        Returns:
            List of ConsensusVote objects from workers.
            Returns empty list if no votes received or on error.

        Note:
            Implementations should be fault-tolerant:
            - Handle network errors gracefully
            - Respect timeouts
            - Never raise exceptions (return empty list instead)
        """
        pass
