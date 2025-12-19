"""
Domain Service: Inference Engine

Handles SLM inference for verification, enhancement, and hallucination detection.
"""

import time
from typing import Dict, Protocol, Optional, Any


class IModelBackend(Protocol):
    """Interface for model backend (transformers, etc.)."""

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text from prompt."""
        ...

    def get_model_id(self) -> str:
        """Get model identifier."""
        ...


class InferenceEngine:
    """
    Domain service for SLM inference.

    Orchestrates prompt building, model inference, and response parsing.
    """

    def __init__(
        self,
        model_backend: IModelBackend,
        prompt_builder: Any,  # PromptBuilder
        response_parser: Any,  # ResponseParser
        max_tokens: int = 150,
    ):
        """
        Initialize inference engine.

        Args:
            model_backend: Backend for model inference
            prompt_builder: Service for building prompts
            response_parser: Service for parsing responses
            max_tokens: Maximum tokens to generate
        """
        self.model_backend = model_backend
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.max_tokens = max_tokens

    def verify_content(
        self,
        query: str,
        content: str,
        shard_summary: Optional[str] = None,
        epoch_summary: Optional[str] = None,
    ) -> Dict:
        """
        Verify if content answers query using SLM.

        Returns:
            Dict with confidence, supports_query, reasoning, model_id, inference_time_ms
        """
        start_time = time.time()

        # Build prompt
        prompt = self.prompt_builder.build_verification_prompt(
            query, content, shard_summary, epoch_summary
        )

        # Generate response
        response_text = self.model_backend.generate(prompt, self.max_tokens)

        # Parse response
        result = self.response_parser.parse_verification_response(response_text)

        # Add metadata
        result["model_id"] = self.model_backend.get_model_id()
        result["inference_time_ms"] = (time.time() - start_time) * 1000

        return result

    def enhance_query(
        self, query: str, timestamp_context: Optional[str] = None
    ) -> Dict:
        """
        Enhance query for better retrieval using SLM.

        Returns:
            Dict with enhanced_query, expansions, model_id, inference_time_ms
        """
        start_time = time.time()

        # Build prompt
        prompt = self.prompt_builder.build_query_enhancement_prompt(
            query, timestamp_context
        )

        # Generate response
        response_text = self.model_backend.generate(prompt, self.max_tokens)

        # Parse response
        result = self.response_parser.parse_enhancement_response(response_text, query)

        # Add metadata
        result["model_id"] = self.model_backend.get_model_id()
        result["inference_time_ms"] = (time.time() - start_time) * 1000
        result["original_query"] = query

        return result

    def check_hallucination(
        self,
        query: str,
        system_response: str,
        ground_truth: dict,
        valid_terms: list[str],
        confidence: float,
    ) -> Dict:
        """
        Check for hallucinations in system response using SLM.

        Returns:
            Dict with has_hallucination, type, severity, explanation, flagged_content, model_id, inference_time_ms
        """
        start_time = time.time()

        # Build prompt
        prompt = self.prompt_builder.build_hallucination_check_prompt(
            query, system_response, ground_truth, valid_terms, confidence
        )

        # Generate response
        response_text = self.model_backend.generate(prompt, self.max_tokens)

        # Parse response
        result = self.response_parser.parse_hallucination_response(response_text)

        # Add metadata
        result["model_id"] = self.model_backend.get_model_id()
        result["inference_time_ms"] = (time.time() - start_time) * 1000

        return result
