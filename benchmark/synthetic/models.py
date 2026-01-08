"""
Data Models for Synthetic History Generation

Contains enums and dataclasses used throughout the synthetic history module.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import Enum


class Perspective(Enum):
    """Stance vectors for multi-agent verification testing"""
    OPTIMIST = "optimist"
    SKEPTIC = "skeptic"
    METHODOLOGIST = "methodologist"
    THEORIST = "theorist"
    EMPIRICIST = "empiricist"


class ClaimType(Enum):
    """Classification for Semantic Quadrant ground truth"""
    CONSENSUS = "consensus"
    DISPUTED = "disputed"
    NOISE = "noise"


@dataclass
class Claim:
    """A single factual claim that can be agreed or disputed"""
    id: str
    content: str
    claim_type: ClaimType
    topic: str
    timestamp: str
    perspective_views: Dict[str, str] = field(default_factory=dict)
    confidence_by_perspective: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d['claim_type'] = self.claim_type.value
        return d


@dataclass
class Event:
    """An episodic event in the research history"""
    id: str
    timestamp: str
    topic: str
    event_type: str
    content: str
    claims: List[str]
    causal_parents: List[str]
    agent_snapshot_id: str
    perspective: Perspective

    def to_dict(self):
        d = asdict(self)
        d['perspective'] = self.perspective.value
        return d


@dataclass
class Query:
    """A benchmark query with ground truth and per-query validation criteria."""
    id: str
    question: str
    query_type: str
    timestamp_context: Optional[str]
    expected_consensus: List[str]
    expected_disputed: List[Dict]
    expected_sources: List[str]
    difficulty: str
    # Per-query validation criteria
    required_terms: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    validation_pattern: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        d.setdefault('required_terms', [])
        d.setdefault('forbidden_terms', [])
        d.setdefault('validation_pattern', None)
        return d


# Real-world terms that indicate hallucination in the synthetic alternate universe
REAL_WORLD_FORBIDDEN_TERMS = frozenset({
    # Financial/business terms (alternate physics has no funding/grants)
    "grant", "funding", "investment", "budget", "capital", "venture",
    "investor", "shareholder", "profit", "revenue",
    # Real physics terms (alternate universe uses different terminology)
    "electron", "proton", "neutron", "quark", "photon", "graviton",
    "higgs", "neutrino", "muon", "gluon", "meson", "hadron",
    "quantum mechanics", "general relativity", "string theory",
    # Real institutions
    "cern", "nasa", "mit", "stanford", "harvard", "cambridge",
    "lhc", "ligo", "fermilab", "slac",
    # Real scientists
    "einstein", "feynman", "hawking", "dirac", "bohr", "heisenberg",
    "schrodinger", "planck", "newton", "maxwell",
    # AI/chatbot hallucination indicators
    "unfortunately", "i cannot", "i don't have", "my knowledge",
    "as an ai", "language model", "training data", "i apologize"
})
