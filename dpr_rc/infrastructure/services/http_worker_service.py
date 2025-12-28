"""
HttpWorkerService

HTTP-based implementation of IWorkerService.
This is the production implementation that calls passive workers over HTTP.
"""

import os
import requests
from typing import List, Optional

from dpr_rc.application.interfaces import IWorkerService
from dpr_rc.models import ConsensusVote
from dpr_rc.logging_utils import StructuredLogger, ComponentType
from dpr_rc.debug_utils import debug_http_worker_call, debug_http_worker_response, debug_log


class HttpWorkerService(IWorkerService):
    """
    HTTP implementation of worker communication service.

    This service calls passive workers directly via HTTP.
    It implements the EXACT same logic as call_workers_via_http in active_agent.py.
    """

    def __init__(
        self,
        worker_urls: Optional[str] = None,
        timeout: float = 30.0,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize HTTP worker service.

        Args:
            worker_urls: Comma-separated list of worker URLs (default from env)
            timeout: Request timeout in seconds
            logger: Optional logger
        """
        self._worker_urls_str = worker_urls or os.getenv("PASSIVE_WORKER_URL", "")
        self._timeout = timeout
        self._logger = logger or StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

    async def gather_votes(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str] = None
    ) -> List[ConsensusVote]:
        """
        Gather votes from passive workers via HTTP.

        This is the EXACT same implementation as call_workers_via_http
        in active_agent.py, ensuring identical behavior.

        Note: This is async for interface compatibility, but makes synchronous
        HTTP calls (same as active_agent.py). Future implementations may use
        aiohttp for true async.
        """
        if not self._worker_urls_str:
            self._logger.logger.warning("PASSIVE_WORKER_URL not set, cannot call workers via HTTP")
            return []

        # Support multiple worker URLs (comma-separated)
        worker_urls = [url.strip() for url in self._worker_urls_str.split(",") if url.strip()]

        all_votes = []

        for worker_url in worker_urls:
            try:
                # Ensure URL has /process_rfi endpoint
                endpoint = worker_url.rstrip("/")
                if not endpoint.endswith("/process_rfi"):
                    endpoint = f"{endpoint}/process_rfi"

                request_payload = {
                    "trace_id": trace_id,
                    "query_text": query_text,
                    "original_query": original_query,
                    "target_shards": target_shards,
                    "timestamp_context": timestamp_context
                }

                # DEBUG: HTTP worker call
                debug_http_worker_call(trace_id, endpoint, request_payload)

                response = requests.post(
                    endpoint,
                    json=request_payload,
                    timeout=self._timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    votes_data = result.get("votes", [])
                    worker_id = result.get("worker_id", "unknown")

                    # DEBUG: HTTP worker response
                    debug_http_worker_response(
                        trace_id, endpoint, len(votes_data),
                        {"worker_id": worker_id, "votes": votes_data, "shards": result.get("shards_queried", [])}
                    )

                    for vote_data in votes_data:
                        try:
                            vote = ConsensusVote(**vote_data)
                            all_votes.append(vote)
                            self._logger.logger.debug(
                                f"Received vote from {worker_id}: confidence={vote.confidence_score:.2f}"
                            )
                        except Exception as e:
                            self._logger.logger.warning(f"Failed to parse vote: {e}")
                else:
                    # DEBUG: Worker error
                    debug_log("ActiveController", f"Worker error: {response.status_code}",
                             {"url": worker_url, "response": response.text[:500]})
                    self._logger.logger.warning(
                        f"Worker {worker_url} returned {response.status_code}: {response.text[:200]}"
                    )

            except requests.exceptions.RequestException as e:
                self._logger.logger.warning(f"Failed to call worker {worker_url}: {e}")

        self._logger.logger.info(f"HTTP workers returned {len(all_votes)} votes from {len(worker_urls)} workers")
        return all_votes
