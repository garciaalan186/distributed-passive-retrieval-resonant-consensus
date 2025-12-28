"""
Domain Service: Verification Service

Handles L2 verification logic using SLM-based semantic verification.
This is pure domain logic with no infrastructure dependencies.
"""

import time
from typing import Dict, Any, Optional, Protocol
from ..entities import VerificationResult


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
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        """
        Initialize verification service.

        Args:
            slm_client: Client for SLM verification calls
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.slm_client = slm_client
        self.max_retries = max_retries
        self.base_delay = base_delay

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
                result = self.slm_client.verify(
                    query=query,
                    content=content[:2000],  # Limit content size
                    shard_summary=shard_summary[:500] if shard_summary else None,
                    epoch_summary=epoch_summary[:500] if epoch_summary else None,
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

        # Calculate heuristic score
        base_score = len(intersection) / len(union)
        length_factor = min(1.0, len(content) / 200.0)
        v_score = (base_score * 0.6 + length_factor * 0.4)

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, v_score))

        return VerificationResult(
            content=content,
            confidence_score=confidence,
            verified=confidence >= 0.3,
            explanation=f"Fallback heuristic (token overlap: {len(intersection)}/{len(union)})",
            depth_penalty=float(depth),
        )
