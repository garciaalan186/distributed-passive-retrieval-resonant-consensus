"""
Infrastructure: HTTP SLM Client

Concrete implementation of ISLMClient using HTTP requests.
"""

import requests
from typing import Dict, Any, Optional
from dpr_rc.domain.passive_agent.services import (
    ISLMClient,
    ServiceUnavailableError,
    RequestError,
)
from dpr_rc.logging_utils import StructuredLogger, ComponentType


class HttpSLMClient:
    """
    HTTP client for SLM service verification.

    Implements ISLMClient interface using requests library.
    """

    def __init__(
        self,
        slm_service_url: str,
        timeout: int = 30,
        worker_id: str = "unknown",
    ):
        """
        Initialize HTTP SLM client.

        Args:
            slm_service_url: Base URL of SLM service
            timeout: Request timeout in seconds
            worker_id: Worker ID for logging
        """
        self.slm_service_url = slm_service_url
        self.timeout = timeout
        self.worker_id = worker_id
        self.logger = StructuredLogger(ComponentType.PASSIVE_WORKER)

    def verify(
        self,
        query: str,
        content: str,
        shard_summary: Optional[str] = None,
        epoch_summary: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify content against query using SLM service.

        Args:
            query: User's query
            content: Retrieved content to verify
            shard_summary: Optional L1 context (shard summary)
            epoch_summary: Optional L2 context (epoch summary)
            trace_id: Trace ID for logging correlation

        Returns:
            Dictionary with:
            - 'confidence' (float): Verification score [0, 1]
            - 'supports_query' (bool): Whether content supports query
            - 'reasoning' (str): Explanation from SLM

        Raises:
            ServiceUnavailableError: When service returns 503
            RequestError: For other errors
        """
        # Build request payload
        verify_request = {
            "query": query,
            "retrieved_content": content,
            "trace_id": trace_id or f"{self.worker_id}_{int(__import__('time').time())}",
        }

        # Add foveated context if provided
        if shard_summary:
            verify_request["shard_summary"] = shard_summary
        if epoch_summary:
            verify_request["epoch_summary"] = epoch_summary

        # Log request
        self.logger.log_message(
            trace_id=verify_request["trace_id"],
            direction="request",
            message_type="slm_verify",
            payload=verify_request,
            metadata={
                "slm_url": self.slm_service_url,
                "worker_id": self.worker_id,
            },
        )

        try:
            response = requests.post(
                f"{self.slm_service_url}/verify",
                json=verify_request,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()

                # Log response
                self.logger.log_message(
                    trace_id=verify_request["trace_id"],
                    direction="response",
                    message_type="slm_verify",
                    payload=result,
                    metadata={
                        "worker_id": self.worker_id,
                        "confidence": result.get("confidence", 0.0),
                    },
                )

                return result

            elif response.status_code == 503:
                # Service unavailable (model loading)
                self.logger.logger.warning(
                    f"SLM service unavailable (503): {response.text}"
                )
                raise ServiceUnavailableError("SLM service not ready")

            else:
                # Other error
                self.logger.logger.error(
                    f"SLM service error {response.status_code}: {response.text}"
                )
                raise RequestError(
                    f"SLM returned {response.status_code}: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            self.logger.logger.error(f"SLM request failed: {e}")
            raise RequestError(f"Request failed: {e}")
