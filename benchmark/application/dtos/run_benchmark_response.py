"""
RunBenchmarkResponse DTO

Encapsulates results from a benchmark run.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunBenchmarkResponse:
    """
    Response from running a benchmark.

    This DTO contains aggregated metrics and metadata from a complete benchmark run.

    Attributes:
        run_id: Unique identifier for this benchmark run
        scale: Scale level that was benchmarked
        total_queries: Total number of queries executed
        dprrc_accuracy: DPR-RC correctness rate (0.0 to 1.0)
        baseline_accuracy: Baseline correctness rate (0.0 to 1.0)
        dprrc_hallucination_rate: DPR-RC hallucination rate (0.0 to 1.0)
        baseline_hallucination_rate: Baseline hallucination rate (0.0 to 1.0)
        mean_latency_dprrc: Mean latency for DPR-RC queries in milliseconds
        mean_latency_baseline: Mean latency for baseline queries in milliseconds
        p95_latency_dprrc: 95th percentile latency for DPR-RC in milliseconds
        p95_latency_baseline: 95th percentile latency for baseline in milliseconds
        report_path: Path to generated report file
        dataset_path: Path to generated dataset file
        error: Optional error message if benchmark failed

    Design Notes:
        - All metrics are pre-aggregated for easy consumption
        - Paths allow callers to access detailed results
        - Success/failure indicated by error field
    """
    run_id: str
    scale: str
    total_queries: int
    dprrc_accuracy: float
    baseline_accuracy: float
    dprrc_hallucination_rate: float
    baseline_hallucination_rate: float
    mean_latency_dprrc: float
    mean_latency_baseline: float
    p95_latency_dprrc: float
    p95_latency_baseline: float
    report_path: str
    dataset_path: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        """Validate response values"""
        # Validate accuracy rates
        for field, value in [
            ('dprrc_accuracy', self.dprrc_accuracy),
            ('baseline_accuracy', self.baseline_accuracy),
            ('dprrc_hallucination_rate', self.dprrc_hallucination_rate),
            ('baseline_hallucination_rate', self.baseline_hallucination_rate),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field} must be between 0.0 and 1.0, got {value}"
                )

        # Validate latencies
        for field, value in [
            ('mean_latency_dprrc', self.mean_latency_dprrc),
            ('mean_latency_baseline', self.mean_latency_baseline),
            ('p95_latency_dprrc', self.p95_latency_dprrc),
            ('p95_latency_baseline', self.p95_latency_baseline),
        ]:
            if value < 0:
                raise ValueError(
                    f"{field} cannot be negative, got {value}"
                )

    @property
    def succeeded(self) -> bool:
        """Check if benchmark succeeded"""
        return self.error is None

    def get_accuracy_improvement(self) -> float:
        """
        Calculate DPR-RC accuracy improvement over baseline.

        Returns:
            Percentage improvement (e.g., 0.15 means 15% improvement)
        """
        if self.baseline_accuracy == 0:
            return float('inf') if self.dprrc_accuracy > 0 else 0.0

        return (self.dprrc_accuracy - self.baseline_accuracy) / self.baseline_accuracy

    def get_hallucination_reduction(self) -> float:
        """
        Calculate hallucination rate reduction from baseline to DPR-RC.

        Returns:
            Percentage reduction (e.g., 0.30 means 30% reduction)
        """
        if self.baseline_hallucination_rate == 0:
            return 0.0

        return (self.baseline_hallucination_rate - self.dprrc_hallucination_rate) / self.baseline_hallucination_rate

    def get_latency_overhead(self) -> float:
        """
        Calculate DPR-RC latency overhead over baseline.

        Returns:
            Percentage overhead (e.g., 0.25 means 25% overhead)
        """
        if self.mean_latency_baseline == 0:
            return float('inf') if self.mean_latency_dprrc > 0 else 0.0

        return (self.mean_latency_dprrc - self.mean_latency_baseline) / self.mean_latency_baseline
