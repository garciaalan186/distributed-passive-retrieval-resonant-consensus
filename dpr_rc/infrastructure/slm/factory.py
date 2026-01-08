"""
Infrastructure: SLM Factory

Dependency injection factory for assembling SLM components.
"""

import os
import threading
import torch
from dpr_rc.domain.slm.services import PromptBuilder, ResponseParser, InferenceEngine
from dpr_rc.infrastructure.slm.backends import TransformersBackend


class SLMFactory:
    """
    Factory for creating SLM Service components.

    Implements dependency injection pattern.
    """

    _inference_engine_instance = None
    _thread_local = threading.local()

    @staticmethod
    def create_inference_engine(
        model_id: str = "Qwen/Qwen2-0.5B-Instruct",
        device: str = "cpu",
        max_tokens: int = 150,
        use_4bit_quantization: bool = False,
    ) -> InferenceEngine:
        """
        Create fully wired InferenceEngine.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ("cpu" or "cuda")
            max_tokens: Maximum tokens to generate
            use_4bit_quantization: Enable 4-bit quantization for memory efficiency

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
            use_4bit_quantization=use_4bit_quantization,
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

        In multi-GPU mode, uses thread-local storage with GPU pinning.
        Otherwise uses singleton pattern.

        Returns:
            Configured InferenceEngine
        """
        # Check if multi-GPU mode is enabled and we have a GPU assignment
        multi_gpu = os.getenv("ENABLE_MULTI_GPU_WORKERS", "false").lower() == "true"

        if multi_gpu and hasattr(SLMFactory._thread_local, 'gpu_id'):
            # Use thread-local GPU-pinned instance
            return SLMFactory.create_for_thread(SLMFactory._thread_local.gpu_id)

        # Fallback to singleton pattern for non-multi-GPU mode
        model_id = os.getenv("SLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
        max_tokens = int(os.getenv("SLM_MAX_TOKENS", "150"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_4bit = os.getenv("SLM_USE_4BIT_QUANTIZATION", "false").lower() == "true"

        return SLMFactory.create_inference_engine(
            model_id=model_id,
            device=device,
            max_tokens=max_tokens,
            use_4bit_quantization=use_4bit,
        )

    @staticmethod
    def create_for_thread(gpu_id: int) -> InferenceEngine:
        """
        Create SLM instance for current thread, pinned to specific GPU.
        Uses thread-local storage to ensure one instance per thread.

        Args:
            gpu_id: GPU device ID (0 or 1)

        Returns:
            Thread-local InferenceEngine instance
        """
        if not hasattr(SLMFactory._thread_local, 'engine'):
            # First call in this thread - create new instance
            device = f"cuda:{gpu_id}"
            model_id = os.getenv("SLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
            max_tokens = int(os.getenv("SLM_MAX_TOKENS", "150"))
            use_4bit = os.getenv("SLM_USE_4BIT_QUANTIZATION", "false").lower() == "true"

            # Create backend with specific device
            prompt_builder = PromptBuilder()
            response_parser = ResponseParser()

            model_backend = TransformersBackend(
                model_id=model_id,
                device=device,
                torch_dtype=torch.float16,
                device_map=None,  # Single GPU per thread (4-bit uses device_map internally)
                use_4bit_quantization=use_4bit,
            )

            # Wire together
            engine = InferenceEngine(
                model_backend=model_backend,
                prompt_builder=prompt_builder,
                response_parser=response_parser,
                max_tokens=max_tokens,
            )

            SLMFactory._thread_local.engine = engine
            SLMFactory._thread_local.gpu_id = gpu_id

        return SLMFactory._thread_local.engine
