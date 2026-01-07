"""
Infrastructure: SLM Factory

Dependency injection factory for assembling SLM components.
"""

import os
import torch
from dpr_rc.domain.slm.services import PromptBuilder, ResponseParser, InferenceEngine
from dpr_rc.infrastructure.slm.backends import TransformersBackend


class SLMFactory:
    """
    Factory for creating SLM Service components.

    Implements dependency injection pattern.
    """

    _inference_engine_instance = None

    @staticmethod
    def create_inference_engine(
        model_id: str = "Qwen/Qwen2-0.5B-Instruct",
        device: str = "cpu",
        max_tokens: int = 150,
    ) -> InferenceEngine:
        """
        Create fully wired InferenceEngine.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ("cpu" or "cuda")
            max_tokens: Maximum tokens to generate

        Returns:
            Configured InferenceEngine instance
        """
        # Singleton check
        if SLMFactory._inference_engine_instance is not None:
             return SLMFactory._inference_engine_instance
        # Domain Services
        prompt_builder = PromptBuilder()
        response_parser = ResponseParser()

        # Infrastructure: Model Backend
        torch_dtype = torch.float16 if device == "cuda" else None
        
        # Dual GPU Optimization: Use Accelerate's device_map="auto" if multiple GPUs
        device_map = None
        if device == "cuda" and torch.cuda.device_count() > 1:
            device_map = "auto"

        model_backend = TransformersBackend(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        # Wire together
        inference_engine = InferenceEngine(
            model_backend=model_backend,
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            max_tokens=max_tokens,
        )

        SLMFactory._inference_engine_instance = inference_engine
        return inference_engine

    @staticmethod
    def create_from_env() -> InferenceEngine:
        """
        Create inference engine from environment variables.

        Returns:
            Configured InferenceEngine
        """
        model_id = os.getenv("SLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
        max_tokens = int(os.getenv("SLM_MAX_TOKENS", "150"))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return SLMFactory.create_inference_engine(
            model_id=model_id,
            device=device,
            max_tokens=max_tokens,
        )
