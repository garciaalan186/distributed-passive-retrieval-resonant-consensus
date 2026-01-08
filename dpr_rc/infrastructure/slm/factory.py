"""
Infrastructure: SLM Factory

Dependency injection factory for assembling SLM components.
Supports both singleton (single-GPU) and pool (multi-GPU) patterns.
"""

import os
import threading
import torch
from typing import Dict, Optional
from dpr_rc.domain.slm.services import PromptBuilder, ResponseParser, InferenceEngine
from dpr_rc.infrastructure.slm.backends import TransformersBackend


class SLMFactory:
    """
    Factory for creating SLM Service components.

    Supports two modes:
    - Singleton mode: Single shared engine (default, for single-GPU)
    - GPU Pool mode: Per-GPU engines for parallel execution (multi-GPU)
    """

    # Singleton instance for single-GPU mode
    _inference_engine_instance: Optional[InferenceEngine] = None

    # GPU-indexed engine pool for multi-GPU mode
    _gpu_engines: Dict[int, InferenceEngine] = {}

    # Thread safety
    _init_lock = threading.Lock()
    _thread_local = threading.local()

    @staticmethod
    def set_gpu_context(gpu_id: int) -> None:
        """
        Set GPU context for current thread.

        In multi-GPU mode, this determines which GPU engine
        will be used for subsequent calls in this thread.

        Args:
            gpu_id: GPU device ID (0, 1, etc.)
        """
        SLMFactory._thread_local.gpu_id = gpu_id

    @staticmethod
    def clear_gpu_context() -> None:
        """Clear GPU context for current thread."""
        if hasattr(SLMFactory._thread_local, 'gpu_id'):
            delattr(SLMFactory._thread_local, 'gpu_id')

    @staticmethod
    def get_engine() -> InferenceEngine:
        """
        Get inference engine based on current context.

        In multi-GPU mode with GPU context set: returns GPU-specific engine
        Otherwise: returns singleton engine

        Returns:
            Configured InferenceEngine instance
        """
        multi_gpu = os.getenv("ENABLE_MULTI_GPU_WORKERS", "false").lower() == "true"

        if multi_gpu and hasattr(SLMFactory._thread_local, 'gpu_id'):
            gpu_id = SLMFactory._thread_local.gpu_id
            return SLMFactory._get_gpu_engine(gpu_id)
        else:
            return SLMFactory._get_singleton_engine()

    @staticmethod
    def _get_gpu_engine(gpu_id: int) -> InferenceEngine:
        """
        Get or create engine for specific GPU.

        Thread-safe lazy initialization of per-GPU engines.

        Args:
            gpu_id: GPU device ID

        Returns:
            InferenceEngine pinned to specified GPU
        """
        if gpu_id not in SLMFactory._gpu_engines:
            with SLMFactory._init_lock:
                # Double-check locking pattern
                if gpu_id not in SLMFactory._gpu_engines:
                    SLMFactory._gpu_engines[gpu_id] = SLMFactory._create_engine_for_gpu(gpu_id)
        return SLMFactory._gpu_engines[gpu_id]

    @staticmethod
    def _get_singleton_engine() -> InferenceEngine:
        """
        Get or create singleton engine.

        Thread-safe lazy initialization of single shared engine.

        Returns:
            Shared InferenceEngine instance
        """
        if SLMFactory._inference_engine_instance is None:
            with SLMFactory._init_lock:
                # Double-check locking pattern
                if SLMFactory._inference_engine_instance is None:
                    SLMFactory._inference_engine_instance = SLMFactory._create_engine()
        return SLMFactory._inference_engine_instance

    @staticmethod
    def _create_engine(device: Optional[str] = None) -> InferenceEngine:
        """
        Create inference engine with environment configuration.

        Args:
            device: Optional device override (e.g., "cuda:0")

        Returns:
            Configured InferenceEngine
        """
        model_id = os.getenv("SLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
        max_tokens = int(os.getenv("SLM_MAX_TOKENS", "150"))
        use_4bit = os.getenv("SLM_USE_4BIT_QUANTIZATION", "false").lower() == "true"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Domain Services
        prompt_builder = PromptBuilder()
        response_parser = ResponseParser()

        # Infrastructure: Model Backend
        torch_dtype = torch.float16 if "cuda" in device else None

        # For single-GPU mode with multiple GPUs, use device_map="auto"
        device_map = None
        if device == "cuda" and torch.cuda.device_count() > 1 and not use_4bit:
            device_map = "auto"

        model_backend = TransformersBackend(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_4bit_quantization=use_4bit,
        )

        return InferenceEngine(
            model_backend=model_backend,
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _create_engine_for_gpu(gpu_id: int) -> InferenceEngine:
        """
        Create inference engine pinned to specific GPU.

        Args:
            gpu_id: GPU device ID

        Returns:
            InferenceEngine pinned to specified GPU
        """
        device = f"cuda:{gpu_id}"
        model_id = os.getenv("SLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        max_tokens = int(os.getenv("SLM_MAX_TOKENS", "150"))
        use_4bit = os.getenv("SLM_USE_4BIT_QUANTIZATION", "false").lower() == "true"

        # Domain Services
        prompt_builder = PromptBuilder()
        response_parser = ResponseParser()

        model_backend = TransformersBackend(
            model_id=model_id,
            device=device,
            torch_dtype=torch.float16,
            device_map=None,  # Explicit single GPU (4-bit uses device_map internally)
            use_4bit_quantization=use_4bit,
        )

        return InferenceEngine(
            model_backend=model_backend,
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            max_tokens=max_tokens,
        )

    # Legacy API for backwards compatibility
    @staticmethod
    def create_inference_engine(
        model_id: str = "Qwen/Qwen2-0.5B-Instruct",
        device: str = "cpu",
        max_tokens: int = 150,
        use_4bit_quantization: bool = False,
    ) -> InferenceEngine:
        """
        Create fully wired InferenceEngine.

        DEPRECATED: Use get_engine() instead for proper multi-GPU support.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ("cpu" or "cuda")
            max_tokens: Maximum tokens to generate
            use_4bit_quantization: Enable 4-bit quantization for memory efficiency

        Returns:
            Configured InferenceEngine instance
        """
        # Use singleton for backwards compatibility
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

        Automatically selects singleton or GPU pool mode based on
        ENABLE_MULTI_GPU_WORKERS and thread-local GPU context.

        Returns:
            Configured InferenceEngine
        """
        return SLMFactory.get_engine()

    @staticmethod
    def create_for_thread(gpu_id: int) -> InferenceEngine:
        """
        Create SLM instance for current thread, pinned to specific GPU.

        DEPRECATED: Use set_gpu_context() + get_engine() instead.

        Args:
            gpu_id: GPU device ID (0 or 1)

        Returns:
            GPU-pinned InferenceEngine instance
        """
        return SLMFactory._get_gpu_engine(gpu_id)

    @staticmethod
    def prewarm_gpu_engines(num_gpus: int = 2) -> None:
        """
        Pre-initialize GPU engines sequentially to avoid CUDA conflicts.

        Loading multiple 4-bit models simultaneously can cause CUDA memory
        access errors. This method loads them one at a time.

        Args:
            num_gpus: Number of GPU engines to initialize
        """
        import torch
        print(f"Pre-warming {num_gpus} GPU engines...")

        for gpu_id in range(num_gpus):
            if gpu_id >= torch.cuda.device_count():
                print(f"  GPU {gpu_id} not available, skipping")
                continue

            if gpu_id not in SLMFactory._gpu_engines:
                print(f"  Loading model on GPU {gpu_id}...")
                # Sequential initialization (thread-safe via _init_lock)
                SLMFactory._get_gpu_engine(gpu_id)
                print(f"  GPU {gpu_id} ready")

        print("GPU engines pre-warmed and ready")

    @staticmethod
    def reset() -> None:
        """
        Reset factory state. Used for testing.
        """
        with SLMFactory._init_lock:
            SLMFactory._inference_engine_instance = None
            SLMFactory._gpu_engines.clear()
            if hasattr(SLMFactory._thread_local, 'gpu_id'):
                delattr(SLMFactory._thread_local, 'gpu_id')
