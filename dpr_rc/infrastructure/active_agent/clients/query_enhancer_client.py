"""
Infrastructure: Query Enhancer Client

HTTP client for SLM query enhancement.
"""

import requests
from typing import Dict, Any, Optional
from dpr_rc.logging_utils import StructuredLogger, ComponentType


class QueryEnhancerClient:
    """HTTP client for query enhancement via SLM service."""

    def __init__(self, slm_service_url: str, timeout: float = 30.0):
        self.slm_service_url = slm_service_url
        self.timeout = timeout
        self.logger = StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

    def enhance(
        self, query_text: str, timestamp_context: Optional[str], trace_id: str
    ) -> Dict[str, Any]:
        """
        Enhance query using SLM service.

        Args:
            query_text: Original query
            timestamp_context: Optional temporal context
            trace_id: Trace ID for logging

        Returns:
            Dict with enhanced_query, expansions, enhancement_used
        """
        try:
            payload = {
                "query": query_text,
                "timestamp_context": timestamp_context or "",
                "trace_id": trace_id,
            }

            # Log request
            self.logger.log_message(
                trace_id=trace_id,
                direction="request",
                message_type="slm_enhance",
                payload=payload,
                metadata={"slm_url": self.slm_service_url},
            )

            response = requests.post(
                f"{self.slm_service_url}/enhance_query",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                result = response.json()

                # Log response
                self.logger.log_message(
                    trace_id=trace_id,
                    direction="response",
                    message_type="slm_enhance",
                    payload=result,
                    metadata={},
                )

                return {
                    "enhanced_query": result.get("enhanced_query", query_text),
                    "expansions": result.get("expansions", []),
                    "enhancement_used": True,
                    "inference_time_ms": result.get("inference_time_ms", 0),
                }
            else:
                self.logger.logger.warning(
                    f"SLM enhancement failed ({response.status_code}), using original query"
                )
                return {
                    "enhanced_query": query_text,
                    "expansions": [],
                    "enhancement_used": False,
                }

        except Exception as e:
            self.logger.logger.warning(f"SLM enhancement error: {e}, using original query")
            return {
                "enhanced_query": query_text,
                "expansions": [],
                "enhancement_used": False,
            }
