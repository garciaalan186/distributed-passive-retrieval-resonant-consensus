"""
Infrastructure: Worker Communicator

Handles communication with passive workers via HTTP and Redis.
"""

import json
import time
import requests
import asyncio
from typing import List, Optional, Any
from dpr_rc.logging_utils import StructuredLogger, ComponentType
from dpr_rc.models import ConsensusVote


class WorkerCommunicator:
    """
    Communicates with passive workers to gather votes.

    Supports both HTTP (for Cloud Run) and Redis (for local dev).
    """

    def __init__(
        self,
        worker_urls: List[str],
        redis_client: Optional[Any] = None,
        use_http: bool = True,
        http_timeout: float = 90.0,
        redis_timeout: float = 5.0,
        rfi_stream: str = "dpr:rfi",
    ):
        """
        Initialize communicator.

        Args:
            worker_urls: List of worker HTTP endpoints
            redis_client: Optional Redis client
            use_http: Prefer HTTP over Redis
            http_timeout: HTTP request timeout
            redis_timeout: Redis vote collection timeout
            rfi_stream: Redis stream name for RFIs
        """
        self.worker_urls = worker_urls
        self.redis_client = redis_client
        self.use_http = use_http
        self.http_timeout = http_timeout
        self.redis_timeout = redis_timeout
        self.rfi_stream = rfi_stream
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
        Gather votes from workers.

        Tries HTTP first if enabled, falls back to Redis if needed.

        Returns:
            List of ConsensusVote objects
        """
        votes = []

        # Try HTTP workers first
        if self.use_http and self.worker_urls:
            votes = self._call_workers_via_http(
                trace_id, query_text, original_query, target_shards, timestamp_context
            )

        # Fall back to Redis if no votes and Redis available
        if not votes and self.redis_client:
            self.logger.logger.info("Falling back to Redis for vote collection")
            votes = asyncio.run(
                self._gather_votes_via_redis(
                    trace_id, query_text, original_query, target_shards, timestamp_context
                )
            )

        return votes

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

    async def _gather_votes_via_redis(
        self,
        trace_id: str,
        query_text: str,
        original_query: str,
        target_shards: List[str],
        timestamp_context: Optional[str],
    ) -> List[ConsensusVote]:
        """Gather votes via Redis Pub/Sub."""
        votes = []

        # Subscribe to response channel
        pubsub = self.redis_client.pubsub()
        response_channel = f"dpr:responses:{trace_id}"
        pubsub.subscribe(response_channel)

        # Small delay to ensure subscription is active
        await asyncio.sleep(0.05)

        # Broadcast RFI
        rfi_payload = {
            "trace_id": trace_id,
            "query_text": query_text,
            "original_query": original_query,
            "target_shards": json.dumps(target_shards),
            "timestamp_context": timestamp_context or "",
        }
        self.redis_client.xadd(self.rfi_stream, rfi_payload)

        # Wait for votes
        start_time = time.time()
        while (time.time() - start_time) < self.redis_timeout:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message and message.get("data"):
                try:
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    vote_data = json.loads(data)
                    votes.append(ConsensusVote(**vote_data))
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.logger.warning(f"Failed to parse vote: {e}")

            await asyncio.sleep(0.05)

        # Cleanup
        pubsub.unsubscribe(response_channel)
        pubsub.close()

        return votes
