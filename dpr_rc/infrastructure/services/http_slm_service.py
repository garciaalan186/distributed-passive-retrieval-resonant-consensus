"""
HttpSLMService

HTTP-based implementation of ISLMService.
This is the production implementation that calls the SLM service over HTTP.
"""

import os
import requests
from typing import Optional, Dict, Any

from dpr_rc.application.interfaces import ISLMService
from dpr_rc.logging_utils import StructuredLogger, ComponentType


class HttpSLMService(ISLMService):
    """
    HTTP implementation of SLM query enhancement service.

    This service calls the SLM microservice over HTTP to enhance queries.
    It's fault-tolerant and falls back to original query on failure.
    """

    def __init__(
        self,
        slm_service_url: Optional[str] = None,
        timeout: float = 5.0,
        enable_enhancement: bool = True,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize HTTP SLM service.

        Args:
            slm_service_url: Base URL of SLM service (default from env)
            timeout: Request timeout in seconds
            enable_enhancement: Whether enhancement is enabled globally
            logger: Optional logger
        """
        self._slm_url = slm_service_url or os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
        self._timeout = timeout
        self._enable_enhancement = enable_enhancement
        self._logger = logger or StructuredLogger(ComponentType.ACTIVE_CONTROLLER)

    def enhance_query(
        self,
        query_text: str,
        timestamp_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance query via HTTP call to SLM service.

        This is the EXACT same implementation as enhance_query_via_slm
        in active_agent.py, ensuring identical behavior.
        """
        if not self._enable_enhancement:
            return {
                "original_query": query_text,
                "enhanced_query": query_text,
                "expansions": [],
                "enhancement_used": False
            }

        try:
            response = requests.post(
                f"{self._slm_url}/enhance_query",
                json={
                    "query": query_text,
                    "timestamp_context": timestamp_context
                },
                timeout=self._timeout
            )

            if response.status_code == 200:
                result = response.json()
                self._logger.logger.info(
                    f"Query enhanced: '{query_text}' -> '{result.get('enhanced_query', query_text)}' "
                    f"(expansions: {result.get('expansions', [])})"
                )
                return {
                    "original_query": query_text,
                    "enhanced_query": result.get("enhanced_query", query_text),
                    "expansions": result.get("expansions", []),
                    "enhancement_used": True,
                    "inference_time_ms": result.get("inference_time_ms", 0)
                }
            else:
                self._logger.logger.warning(
                    f"SLM enhance_query returned {response.status_code}, using original query"
                )

        except requests.exceptions.RequestException as e:
            self._logger.logger.warning(f"SLM service unavailable for query enhancement: {e}")

        # Fallback: return original query
        return {
            "original_query": query_text,
            "enhanced_query": query_text,
            "expansions": [],
            "enhancement_used": False
        }
