"""
Infrastructure: Worker Communicator

Handles communication with passive workers via HTTP.
"""

import requests
from typing import List, Optional, Any
from dpr_rc.logging_utils import StructuredLogger, ComponentType
from dpr_rc.models import ConsensusVote


class WorkerCommunicator:
    """
    Communicates with passive workers to gather votes via HTTP.
    """

    def __init__(
        self,
        worker_urls: List[str],
        http_timeout: float = 90.0,
    ):
        """
        Initialize communicator.

        Args:
            worker_urls: List of worker HTTP endpoints
            http_timeout: HTTP request timeout
        """
        self.worker_urls = worker_urls
        self.http_timeout = http_timeout
        self.logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

    def gather_votes(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str],
    ) -> List[Any]:
        """
        Gather votes from workers via HTTP.

        Returns:
            List of ConsensusVote objects
        """
        return self._call_workers_via_http(
            trace_id, query_text, original_query, target_shards, timestamp_context
        )

    def _call_workers_via_http(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str],
    ) -> List[ConsensusVote]:
        """Call workers via HTTP."""
        all_votes = []

        for worker_url in self.worker_urls:
            try:
                endpoint = worker_url.rstrip("/")
                if not endpoint.endswith("/process_rfi"):
                    endpoint = f"{endpoint}/process_rfi"

                request_payload = {
                    "trace_id": trace_id,
                    "query_text": query_text,
                    "original_query": original_query,
                    "target_shards": target_shards,
                    "timestamp_context": timestamp_context,
                }

                # Log request
                self.logger.log_message(
                    trace_id=trace_id,
                    direction="request",
                    message_type="worker_rfi",
                    payload=request_payload,
                    metadata={"worker_url": endpoint},
                )

                response = requests.post(
                    endpoint, json=request_payload, timeout=self.http_timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    votes_data = result.get("votes", [])

                    # Log response
                    self.logger.log_message(
                        trace_id=trace_id,
                        direction="response",
                        message_type="worker_votes",
                        payload=result,
                        metadata={"vote_count": len(votes_data)},
                    )

                    for vote_data in votes_data:
                        try:
                            vote = ConsensusVote(**vote_data)
                            all_votes.append(vote)
                        except Exception as e:
                            self.logger.logger.warning(f"Failed to parse vote: {e}")
                else:
                    self.logger.logger.warning(
                        f"Worker {worker_url} returned {response.status_code}"
                    )

            except requests.exceptions.RequestException as e:
                self.logger.logger.warning(f"Failed to call worker {worker_url}: {e}")

        return all_votes
