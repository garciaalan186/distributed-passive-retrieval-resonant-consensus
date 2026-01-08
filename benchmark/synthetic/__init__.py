"""
Synthetic History Generation Module

Provides phonotactic noun generation and alternate universe physics
to create benchmark datasets with zero real-world knowledge overlap.
"""

from benchmark.synthetic.models import (
    Perspective,
    ClaimType,
    Claim,
    Event,
    Query,
    REAL_WORLD_FORBIDDEN_TERMS,
)
from benchmark.synthetic.phonotactic import PhonotacticGenerator
from benchmark.synthetic.physics import AlternatePhysics
from benchmark.synthetic.domain import AlternateResearchDomain
from benchmark.synthetic.generator import SyntheticHistoryGeneratorV2

__all__ = [
    "Perspective",
    "ClaimType",
    "Claim",
    "Event",
    "Query",
    "REAL_WORLD_FORBIDDEN_TERMS",
    "PhonotacticGenerator",
    "AlternatePhysics",
    "AlternateResearchDomain",
    "SyntheticHistoryGeneratorV2",
]
