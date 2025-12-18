"""
Tempo-normalized segmentation module for DPR-RC.

This module implements intelligent time shard boundary detection using:
- Tempo-normalized inter-event intervals (IEI)
- Lexical coherence analysis
- Information density constraints (H_max)
- Causal chain preservation

Based on the tempo-normalized segmentation paper and DPR-RC Mathematical Model Section 8.
"""

from .tempo_normalizer import TempoNormalizer, BoundaryCandidate
from .coherence_analyzer import CoherenceAnalyzer
from .boundary_detector import BoundaryDetector
from .density_optimizer import DensityOptimizer, CORRECTED_MAX_SHARD_TOKENS
from .causal_aware_partitioner import CausalAwarePartitioner

__all__ = [
    'TempoNormalizer',
    'BoundaryCandidate',
    'CoherenceAnalyzer',
    'BoundaryDetector',
    'DensityOptimizer',
    'CausalAwarePartitioner',
    'CORRECTED_MAX_SHARD_TOKENS',
]
