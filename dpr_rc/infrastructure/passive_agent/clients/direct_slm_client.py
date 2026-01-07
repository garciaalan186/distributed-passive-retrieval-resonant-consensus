"""
Infrastructure: Direct SLM Client

Concrete implementation of ISLMClient using DIRECT in-process calls.
Used for local optimized benchmarking to bypass HTTP overhead.
"""

from typing import Dict, Any, Optional
from dpr_rc.domain.passive_agent.services import (
    ISLMClient,
    RequestError,
)
from dpr_rc.infrastructure.slm import SLMFactory
from dpr_rc.logging_utils import StructuredLogger, ComponentType
import time


class DirectSLMClient:
    """
    Direct client for SLM service verification.

    Implements ISLMClient interface using in-process SLM Engine.
    """

    def __init__(
        self,
        worker_id: str = "unknown",
        slm_service_url: str = None, # Ignored, but kept for signature compatibility
        timeout: int = 30, # Ignored
    ):
        """
        Initialize Direct SLM client.
        """
        self.worker_id = worker_id
        self.logger = StructuredLogger(ComponentType.PASSIVE_WORKER)
        
        # Lazy load engine
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self.logger.logger.info("Initializing Direct SLM Engine...")
            self._engine = SLMFactory.create_from_env()
        return self._engine

    def verify(
        self,
        query: str,
        content: str,
        shard_summary: Optional[str] = None,
        epoch_summary: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify content against query using in-process SLM Engine.
        """
        # Build request payload for logging
        verify_request = {
            "query": query,
            "retrieved_content": content,
            "trace_id": trace_id or f"{self.worker_id}_{int(time.time())}",
        }

        # Log request
        self.logger.log_message(
            trace_id=verify_request["trace_id"],
            direction="request",
            message_type="slm_verify",
            payload=verify_request,
            metadata={
                "slm_mode": "DIRECT",
                "worker_id": self.worker_id,
            },
        )

        try:
            # Call engine directly
            result = self.engine.verify_content(
                query=query,
                content=content,
                shard_summary=shard_summary,
                epoch_summary=epoch_summary
            )

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

        except Exception as e:
            self.logger.logger.error(f"Direct SLM verify failed: {e}")
            raise RequestError(f"Direct verification failed: {e}")
