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


class RevisionType(Enum):
    """Classification for belief revision events"""
    DISPROVEN = "disproven"      # Original claim found to be false
    REFINED = "refined"          # Original claim updated with more precision
    SUPERSEDED = "superseded"    # Original claim replaced by better model


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


@dataclass
class RevisionEvent:
    """
    A belief revision event that contradicts/updates an earlier claim.

    Used for testing temporal RAG systems' ability to:
    1. Return current (not outdated) information
    2. Correctly attribute beliefs to their temporal context
    3. Avoid hallucinations from superseded claims
    """
    id: str
    original_claim_id: str          # ID of the claim being revised
    original_claim_content: str     # Content of the original claim (for reference)
    original_year: int              # When original claim was made
    revision_year: int              # When revision occurred
    revision_type: RevisionType     # disproven, refined, or superseded
    content: str                    # Event text describing the revision
    domain: str                     # Research domain
    superseding_claim_id: Optional[str] = None  # New claim (if any)
    superseding_claim_content: Optional[str] = None  # New claim content

    def to_dict(self):
        d = asdict(self)
        d['revision_type'] = self.revision_type.value
        return d

    def to_event(self) -> Event:
        """Convert to standard Event for ingestion into ChromaDB."""
        return Event(
            id=self.id,
            timestamp=f"{self.revision_year}-06-15T12:00:00Z",
            topic=self.domain,
            event_type="revision",
            content=self.content,
            claims=[self.original_claim_id],
            causal_parents=[],
            agent_snapshot_id=f"revision_{self.id[:4]}",
            perspective=Perspective.METHODOLOGIST  # Revisions are methodological
        )


@dataclass
class RevisionMetadata:
    """
    Metadata tracking all revisions for validation purposes.

    Used by benchmark validation to check if responses correctly
    handle revised beliefs.
    """
    # Maps claim_id -> list of revision events that invalidate it
    disproven_claims: Dict[str, List[str]] = field(default_factory=dict)

    # Maps year -> list of claim_ids valid at that year
    claims_valid_at_year: Dict[int, List[str]] = field(default_factory=dict)

    # Maps claim_id -> superseding claim_id (if any)
    supersession_chain: Dict[str, str] = field(default_factory=dict)

    # All revision events
    revision_events: List[RevisionEvent] = field(default_factory=list)

    def to_dict(self):
        return {
            "disproven_claims": self.disproven_claims,
            "claims_valid_at_year": {str(k): v for k, v in self.claims_valid_at_year.items()},
            "supersession_chain": self.supersession_chain,
            "revision_events": [r.to_dict() for r in self.revision_events]
        }


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
