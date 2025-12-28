"""
RunBenchmarkRequest DTO

Encapsulates all parameters needed to run a benchmark.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunBenchmarkRequest:
    """
    Request to run a benchmark.

    This DTO contains all configuration needed to execute a full benchmark run,
    including scale, executors, and optional features.

    Attributes:
        scale: Scale level ('small', 'medium', 'large', 'stress')
        output_dir: Directory to save benchmark results
        enable_hallucination_detection: Whether to use SLM for hallucination detection
        slm_service_url: URL for SLM service (required if hallucination detection enabled)
        seed: Optional random seed for reproducible dataset generation

    Design Notes:
        - Executors are NOT included in the request; they're injected via the use case
        - This keeps the request focused on what to benchmark, not how to execute
        - The use case is responsible for wiring up executors (via factory)
    """
    scale: str
    output_dir: str
    enable_hallucination_detection: bool = True
    slm_service_url: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate request parameters"""
        valid_scales = ['small', 'medium', 'large', 'stress']
        if self.scale not in valid_scales:
            raise ValueError(
                f"Invalid scale '{self.scale}'. Must be one of: {valid_scales}"
            )

        if self.enable_hallucination_detection and not self.slm_service_url:
            raise ValueError(
                "slm_service_url is required when hallucination detection is enabled"
            )

        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
