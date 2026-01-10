"""
Domain Service: Verification Service

Handles L2 verification logic using SLM-based semantic verification.
This is pure domain logic with no infrastructure dependencies.
"""

import time
from typing import Dict, Any, Optional, Protocol
from ..entities import VerificationResult
from dpr_rc.config import get_dpr_config

# Load verification config
_verification_config = get_dpr_config().get('verification', {})


class ISLMClient(Protocol):
    """Interface for SLM verification client."""

    def verify(
        self,
        query: str,
        content: str,
        shard_summary: Optional[str] = None,
        epoch_summary: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify content against query using SLM reasoning.

        Returns:
            Dictionary with 'confidence' (float), 'supports_query' (bool), 'reasoning' (str)

        Raises:
            ServiceUnavailableError: When SLM service is not ready (503)
            RequestError: For other HTTP errors or network issues
        """
        ...


class ServiceUnavailableError(Exception):
    """Raised when SLM service is temporarily unavailable."""
    pass


class RequestError(Exception):
    """Raised when SLM request fails."""
    pass


class VerificationService:
    """
    Domain service for L2 verification.

    Implements RCP v4 Equation 9:
    C(r_p) = V(q, context_p) · (1 / (1 + i))

    Where:
    - V is semantic verification score from SLM
    - i is hierarchy depth
    """

    def __init__(
        self,
        slm_client: ISLMClient,
        max_retries: int = None,
        base_delay: float = None,
    ):
        """
        Initialize verification service.

        Args:
            slm_client: Client for SLM verification calls
            max_retries: Maximum number of retry attempts (default from config)
            base_delay: Base delay for exponential backoff in seconds (default from config)
        """
        slm_service_config = get_dpr_config().get('slm', {}).get('service', {})
        self.slm_client = slm_client
        self.max_retries = max_retries if max_retries is not None else slm_service_config.get('max_retries', 3)
        self.base_delay = base_delay if base_delay is not None else slm_service_config.get('base_delay', 2.0)

    def verify(
        self,
        query: str,
        content: str,
        depth: int = 0,
        foveated_context: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify content against query with hierarchical foveated context.

        Args:
            query: User's query (original, not enhanced)
            content: Retrieved content to verify
            depth: Hierarchy depth for penalty calculation
            foveated_context: Optional dict with 'L1' (shard summary), 'L2' (epoch summary)
            trace_id: Trace ID for logging

        Returns:
            VerificationResult with adjusted confidence
        """
        # Extract foveated context
        shard_summary = None
        epoch_summary = None
        if foveated_context:
            shard_summary = foveated_context.get("L1")
            epoch_summary = foveated_context.get("L2")

        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            if attempt > 0:
                delay = self.base_delay ** attempt  # Exponential: 2s, 4s, 8s
                time.sleep(delay)

            try:
                content_limit = _verification_config.get('content_limit', 2000)
                summary_limit = _verification_config.get('summary_limit', 500)
                result = self.slm_client.verify(
                    query=query,
                    content=content[:content_limit],  # Limit content size
                    shard_summary=shard_summary[:summary_limit] if shard_summary else None,
                    epoch_summary=epoch_summary[:summary_limit] if epoch_summary else None,
                    trace_id=trace_id,
                )

                base_score = result.get("confidence", 0.5)
                verified = result.get("supports_query", False)
                reasoning = result.get("reasoning", "")

                # Apply depth penalty per RCP v4 Eq. 9
                return VerificationResult(
                    content=content,
                    confidence_score=base_score,
                    verified=verified,
                    explanation=reasoning,
                    depth_penalty=float(depth),
                )

            except ServiceUnavailableError:
                # Service still loading, retry
                if attempt == self.max_retries - 1:
                    # Last attempt failed, fall back
                    break
                continue

            except RequestError:
                # Other error, don't retry
                break

        # Fallback to heuristic verification
        return self._verify_fallback(query, content, depth)

    def _verify_fallback(
        self, query: str, content: str, depth: int = 0
    ) -> VerificationResult:
        """
        Fallback verification using token overlap heuristic.

        Used only when SLM service is unavailable after all retries.
        """
        query_tokens = set(query.lower().split())
        content_tokens = set(content.lower().split())

        if not query_tokens:
            return VerificationResult(
                content=content,
                confidence_score=0.0,
                verified=False,
                explanation="Empty query",
                depth_penalty=float(depth),
            )

        intersection = query_tokens & content_tokens
        union = query_tokens | content_tokens

        if not union:
            return VerificationResult(
                content=content,
                confidence_score=0.0,
                verified=False,
                explanation="No token overlap",
                depth_penalty=float(depth),
            )

        # Calculate heuristic score using config weights
        weights = _verification_config.get('weights', {})
        token_weight = weights.get('token_overlap', 0.6)
        length_weight = weights.get('length_factor', 0.4)
        length_ref = _verification_config.get('length_reference', 200.0)

        base_score = len(intersection) / len(union)
        length_factor = min(1.0, len(content) / length_ref)
        v_score = (base_score * token_weight + length_factor * length_weight)

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, v_score))
        fallback_threshold = _verification_config.get('fallback_threshold', 0.3)

        return VerificationResult(
            content=content,
            confidence_score=confidence,
            verified=confidence >= fallback_threshold,
            explanation=f"Fallback heuristic (token overlap: {len(intersection)}/{len(union)})",
            depth_penalty=float(depth),
        )
