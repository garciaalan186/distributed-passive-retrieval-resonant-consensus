"""
Benchmark Data Models

Dataclasses for benchmark evaluation results.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SuperpositionEvaluation:
    """Evaluation for DPR-RC superposition responses"""
    correct_answer_present: bool
    correct_in_consensus: bool
    correct_in_perspectival: bool
    ideal_placement: bool
    multiple_alternatives_presented: bool


@dataclass
class HallucinationAnalysis:
    """Analysis results for hallucination detection"""
    total_claims: int
    hallucinated_claims: int
    hallucination_rate: float
    legitimate_alternatives: int
    details: List[Dict]


@dataclass
class ResourceMetrics:
    """Resource usage metrics for benchmark queries"""
    mean_latency_ms: float
    p95_latency_ms: float
    tokens_per_query: float
    estimated_cost_usd: float
