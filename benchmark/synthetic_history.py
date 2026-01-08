"""
Synthetic History Generator v2 for DPR-RC Benchmarking

KEY IMPROVEMENT: Uses phonotactic nouns and alternate universe physics
to eliminate prior knowledge confounds in evaluation.

Why this matters:
- Real topics (Quantum Computing, Fusion) are already in model embeddings
- LLMs have parametric knowledge that could leak into retrieval evaluation
- Using novel terms ensures success is due to architecture, not memorization

Approach:
- Phonotactic noun generator (pronounceable nonsense following English phonology)
- Alternate universe physics with consistent internal logic
- Preserves structural complexity while eliminating semantic contamination
"""

import json
import random
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import itertools
import re


# =============================================================================
# PHONOTACTIC NOUN GENERATOR
# =============================================================================

class PhonotacticGenerator:
    """
    Generates pronounceable nonsense words following English phonotactics.
    
    Ensures:
    - Words are pronounceable (valid onset-nucleus-coda patterns)
    - No collision with real English words
    - Consistent generation (seeded)
    - Distinct "feel" for different semantic categories
    """
    
    # English phonotactic patterns
    ONSETS = [
        '', 'b', 'bl', 'br', 'ch', 'd', 'dr', 'f', 'fl', 'fr', 'g', 'gl', 'gr',
        'h', 'j', 'k', 'kl', 'kr', 'l', 'm', 'n', 'p', 'pl', 'pr', 'qu', 'r',
        's', 'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'spl', 'spr', 'st', 'str', 'sw',
        't', 'th', 'tr', 'tw', 'v', 'w', 'wh', 'wr', 'z'
    ]
    
    NUCLEI = ['a', 'e', 'i', 'o', 'u', 'ai', 'au', 'ea', 'ee', 'ei', 'ie', 'oa', 'oo', 'ou']
    
    CODAS = [
        '', 'b', 'ch', 'd', 'f', 'g', 'k', 'l', 'lk', 'lm', 'lp', 'lt', 'm', 'mp',
        'n', 'nd', 'ng', 'nk', 'nt', 'p', 'r', 'rb', 'rd', 'rf', 'rg', 'rk', 'rl',
        'rm', 'rn', 'rp', 'rs', 'rt', 's', 'sh', 'sk', 'sp', 'st', 't', 'th', 'x', 'z'
    ]
    
    # Category-specific phonetic biases (different "feel" for different domains)
    CATEGORY_BIASES = {
        'field': {  # Research fields - longer, more "scientific" sounding
            'preferred_onsets': ['th', 'kr', 'gl', 'str', 'v', 'z'],
            'preferred_nuclei': ['o', 'i', 'a', 'ou', 'ei'],
            'preferred_codas': ['m', 'n', 'x', 'th', 'rs'],
            'syllables': (2, 4)
        },
        'particle': {  # Fundamental particles - short, punchy
            'preferred_onsets': ['k', 'qu', 'b', 'g', 'z'],
            'preferred_nuclei': ['a', 'o', 'u', 'i'],
            'preferred_codas': ['k', 'n', 't', 'x', ''],
            'syllables': (1, 2)
        },
        'unit': {  # Units of measurement - technical sounding
            'preferred_onsets': ['m', 'k', 'j', 'v', 'w'],
            'preferred_nuclei': ['e', 'o', 'a', 'ou'],
            'preferred_codas': ['l', 't', 'n', 's', 'r'],
            'syllables': (1, 2)
        },
        'process': {  # Processes/phenomena - flowing, verb-like
            'preferred_onsets': ['fl', 'v', 'r', 'sw', 'gl'],
            'preferred_nuclei': ['a', 'o', 'i', 'ea', 'ou'],
            'preferred_codas': ['ng', 'n', 'tion', 'm', 'sh'],
            'syllables': (2, 3)
        },
        'entity': {  # Generic entities - neutral
            'preferred_onsets': ['b', 'l', 'm', 'n', 'p', 'r', 's', 't'],
            'preferred_nuclei': ['a', 'e', 'i', 'o', 'u'],
            'preferred_codas': ['', 'd', 'n', 'r', 's', 't'],
            'syllables': (2, 3)
        }
    }
    
    # Words to avoid (common English words that might be generated)
    BLACKLIST = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
        'boy', 'did', 'man', 'she', 'too', 'use', 'war', 'big', 'god', 'lot'
    }
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.generated = set()  # Track to avoid duplicates
        
    def _generate_syllable(self, category: str = 'entity') -> str:
        """Generate a single syllable with category-appropriate phonetics"""
        bias = self.CATEGORY_BIASES.get(category, self.CATEGORY_BIASES['entity'])
        
        # 60% chance to use preferred sounds, 40% any valid sound
        if self.rng.random() < 0.6:
            onset = self.rng.choice(bias['preferred_onsets'])
            nucleus = self.rng.choice(bias['preferred_nuclei'])
            coda = self.rng.choice(bias['preferred_codas'])
        else:
            onset = self.rng.choice(self.ONSETS)
            nucleus = self.rng.choice(self.NUCLEI)
            coda = self.rng.choice(self.CODAS)
            
        return onset + nucleus + coda
    
    def generate_word(self, category: str = 'entity', min_syllables: int = None, 
                      max_syllables: int = None) -> str:
        """Generate a phonotactically valid nonsense word"""
        bias = self.CATEGORY_BIASES.get(category, self.CATEGORY_BIASES['entity'])
        
        if min_syllables is None:
            min_syllables = bias['syllables'][0]
        if max_syllables is None:
            max_syllables = bias['syllables'][1]
            
        # Try to generate unique word
        for _ in range(100):
            num_syllables = self.rng.randint(min_syllables, max_syllables)
            word = ''.join(self._generate_syllable(category) for _ in range(num_syllables))
            
            # Clean up double letters and awkward combinations
            word = re.sub(r'(.)\1{2,}', r'\1\1', word)  # Max 2 repeated chars
            
            # Check validity
            if (len(word) >= 3 and 
                word not in self.BLACKLIST and 
                word not in self.generated):
                self.generated.add(word)
                return word.capitalize()
        
        # Fallback: add random suffix
        base = self._generate_syllable(category)
        return (base + str(self.rng.randint(1, 99))).capitalize()
    
    def generate_compound(self, category: str = 'field') -> str:
        """Generate a two-word compound term"""
        adj = self.generate_word('entity', 2, 3)
        noun = self.generate_word(category, 2, 3)
        return f"{adj} {noun}"


# =============================================================================
# ALTERNATE UNIVERSE PHYSICS
# =============================================================================

class AlternatePhysics:
    """
    Generates a self-consistent alternate universe physics system.
    
    Creates:
    - Fundamental particles with different names/properties
    - Physical constants with different values
    - Phenomena that follow internal logic but differ from our universe
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.phono = PhonotacticGenerator(seed)
        
        # Generate fundamental particles
        self.particles = self._generate_particles()
        
        # Generate physical constants
        self.constants = self._generate_constants()
        
        # Generate phenomena
        self.phenomena = self._generate_phenomena()
        
        # Generate units
        self.units = self._generate_units()
        
    def _generate_particles(self) -> Dict[str, Dict]:
        """Generate alternate universe fundamental particles"""
        particles = {}
        
        # Our universe analogs (but different names/properties)
        particle_types = [
            ('fermion', 6),   # Like quarks
            ('boson', 4),     # Like force carriers
            ('lepton', 4),    # Like electrons/neutrinos
            ('exotic', 3)     # No analog - truly novel
        ]
        
        for ptype, count in particle_types:
            for i in range(count):
                name = self.phono.generate_word('particle')
                particles[name] = {
                    'type': ptype,
                    'charge': self.rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2]),
                    'spin': self.rng.choice([0, 0.5, 1, 1.5, 2]),
                    'mass_unit': self.rng.uniform(0.001, 1000),
                    'stability': self.rng.choice(['stable', 'metastable', 'unstable', 'virtual']),
                    'discovered_year': self.rng.randint(2015, 2025)
                }
        
        return particles
    
    def _generate_constants(self) -> Dict[str, Dict]:
        """Generate alternate physical constants"""
        constants = {}
        
        constant_templates = [
            ('speed_limit', 'velocity', 1e6, 1e10),      # Like c
            ('coupling', 'dimensionless', 0.001, 0.1),   # Like fine structure
            ('quantum', 'action', 1e-35, 1e-32),         # Like h
            ('field_strength', 'force', 1e-12, 1e-8),    # Like G
            ('entropy_base', 'temperature', 1e-24, 1e-20) # Like k
        ]
        
        for template, dim, low, high in constant_templates:
            name = self.phono.generate_word('unit')
            constants[name] = {
                'type': template,
                'dimension': dim,
                'value': self.rng.uniform(low, high),
                'unit': self.phono.generate_word('unit', 1, 2),
                'uncertainty': self.rng.uniform(0.0001, 0.01)
            }
        
        return constants
    
    def _generate_phenomena(self) -> Dict[str, Dict]:
        """Generate alternate physical phenomena"""
        phenomena = {}
        
        phenomenon_types = [
            'field_effect',      # Like electromagnetism
            'quantum_effect',    # Like superposition
            'thermodynamic',     # Like entropy
            'relativistic',      # Like time dilation
            'emergent'           # Like superconductivity
        ]
        
        for ptype in phenomenon_types:
            for i in range(3):
                name = self.phono.generate_word('process')
                
                # Select particles involved
                involved_particles = self.rng.sample(
                    list(self.particles.keys()), 
                    min(3, len(self.particles))
                )
                
                phenomena[name] = {
                    'type': ptype,
                    'particles_involved': involved_particles,
                    'energy_scale': self.rng.choice(['low', 'medium', 'high', 'extreme']),
                    'reversible': self.rng.choice([True, False]),
                    'discovered_year': self.rng.randint(2015, 2025)
                }
        
        return phenomena
    
    def _generate_units(self) -> Dict[str, str]:
        """Generate alternate measurement units"""
        dimensions = [
            'length', 'time', 'mass', 'charge', 'temperature',
            'energy', 'momentum', 'field_strength', 'entropy'
        ]
        
        return {dim: self.phono.generate_word('unit', 1, 2) for dim in dimensions}


# =============================================================================
# RESEARCH DOMAIN GENERATOR (ALTERNATE UNIVERSE)
# =============================================================================

class AlternateResearchDomain:
    """
    A research domain in the alternate universe with self-consistent terminology.
    """
    
    def __init__(self, name: str, physics: AlternatePhysics, seed: int = 42):
        self.name = name
        self.physics = physics
        self.rng = random.Random(seed)
        self.phono = PhonotacticGenerator(seed + hash(name) % 10000)
        
        # Domain-specific terminology
        self.key_concepts = self._generate_concepts()
        self.milestones = self._generate_milestones()
        self.metrics = self._generate_metrics()
        
    def _generate_concepts(self) -> Dict[str, str]:
        """Generate key concepts for this domain"""
        concepts = {}
        concept_types = [
            'fundamental_object',
            'measurement_technique', 
            'theoretical_framework',
            'experimental_apparatus',
            'computational_method'
        ]
        
        for ctype in concept_types:
            for i in range(3):
                name = self.phono.generate_word('field')
                
                # Link to physics
                if self.physics.particles:
                    related_particle = self.rng.choice(list(self.physics.particles.keys()))
                else:
                    related_particle = "unknown"
                    
                concepts[name] = {
                    'type': ctype,
                    'related_particle': related_particle,
                    'complexity': self.rng.choice(['basic', 'intermediate', 'advanced']),
                    'introduced_year': self.rng.randint(2015, 2020)
                }
        
        return concepts
    
    def _generate_milestones(self) -> Dict[int, str]:
        """Generate research milestones timeline"""
        milestones = {}
        
        # Generate milestone descriptions using alternate terminology
        stages = [
            "theoretical {} framework proposed",
            "first {} detection reported",
            "{} manipulation demonstrated",
            "{} coherence achieved",
            "{} scaling breakthrough",
            "practical {} applications",
            "{} efficiency threshold crossed",
            "{} integration with {} systems",
            "{} error correction solved",
            "commercial {} prototypes",
            "{} technology standardization"
        ]
        
        concepts = list(self.key_concepts.keys())
        
        for i, year in enumerate(range(2015, 2026)):
            template = stages[i % len(stages)]
            
            # Fill in with domain concepts
            if '{}' in template:
                num_slots = template.count('{}')
                fillers = [self.rng.choice(concepts) for _ in range(num_slots)]
                milestone = template.format(*fillers)
            else:
                milestone = template
                
            milestones[year] = milestone
            
        return milestones
    
    def _generate_metrics(self) -> Dict[str, Dict]:
        """Generate domain-specific metrics"""
        metrics = {}
        
        for i in range(5):
            name = self.phono.generate_word('unit')
            metrics[name] = {
                'unit': self.rng.choice(list(self.physics.units.values())),
                'baseline_2015': self.rng.uniform(0.01, 1.0),
                'target_2025': self.rng.uniform(10, 1000),
                'yearly_improvement': self.rng.uniform(1.2, 2.0)  # Multiplier
            }
        
        return metrics


# =============================================================================
# MAIN GENERATOR (UPDATED)
# =============================================================================

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
    """A benchmark query with ground truth"""
    id: str
    question: str
    query_type: str
    timestamp_context: Optional[str]
    expected_consensus: List[str]
    expected_disputed: List[Dict]
    expected_sources: List[str]
    difficulty: str
    
    def to_dict(self):
        return asdict(self)


class SyntheticHistoryGeneratorV2:
    """
    Generates large-scale synthetic research history using phonotactic nouns
    and alternate universe physics to eliminate prior knowledge confounds.
    """
    
    def __init__(
        self,
        start_year: int = 2015,
        end_year: int = 2025,
        events_per_topic_per_year: int = 100,
        perspectives_per_event: int = 3,
        num_domains: int = 4,
        seed: int = 42
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.events_per_topic_per_year = events_per_topic_per_year
        self.perspectives_per_event = perspectives_per_event
        self.rng = random.Random(seed)
        
        # Create alternate universe physics
        print("Generating alternate universe physics...")
        self.physics = AlternatePhysics(seed)
        
        # Create research domains with alternate terminology
        print(f"Generating {num_domains} research domains...")
        self.phono = PhonotacticGenerator(seed + 1000)
        self.domains = [
            AlternateResearchDomain(
                self.phono.generate_compound('field'),
                self.physics,
                seed + i
            )
            for i in range(num_domains)
        ]
        
        # Event type distributions
        self.event_types = [
            ("experiment", 0.3),
            ("publication", 0.25),
            ("meeting", 0.15),
            ("discovery", 0.1),
            ("revision", 0.1),
            ("collaboration", 0.05),
            ("funding", 0.05)
        ]
        
        # Storage
        self.events: List[Event] = []
        self.claims: Dict[str, Claim] = {}
        self.queries: List[Query] = []
        self.causal_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Terminology glossary for evaluation
        self.glossary = self._build_glossary()
        
    def _build_glossary(self) -> Dict:
        """Build glossary of all generated terms for reference"""
        return {
            'physics': {
                'particles': self.physics.particles,
                'constants': self.physics.constants,
                'phenomena': self.physics.phenomena,
                'units': self.physics.units
            },
            'domains': {
                d.name: {
                    'concepts': d.key_concepts,
                    'milestones': d.milestones,
                    'metrics': d.metrics
                }
                for d in self.domains
            }
        }
        
    def _generate_timestamp(self, year: int) -> str:
        """Generate random timestamp within a year"""
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        hour = self.rng.randint(8, 18)
        minute = self.rng.randint(0, 59)
        return f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"
    
    def _select_event_type(self) -> str:
        """Weighted random selection of event type"""
        types, weights = zip(*self.event_types)
        return self.rng.choices(types, weights=weights)[0]
    
    def _generate_claim_id(self, content: str) -> str:
        """Deterministic claim ID from content"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_event_content(
        self,
        domain: AlternateResearchDomain,
        year: int,
        event_type: str,
        perspective: Perspective
    ) -> str:
        """Generate event content using alternate universe terminology"""
        
        milestone = domain.milestones.get(year, "ongoing research")
        concepts = list(domain.key_concepts.keys())
        metrics = list(domain.metrics.keys())
        particles = list(self.physics.particles.keys())
        phenomena = list(self.physics.phenomena.keys())
        
        # Templates using alternate terminology with explicit status language
        # Each template explicitly mentions "status" to answer queries like
        # "What was the status of {domain} research in {year}?"
        templates = {
            "experiment": [
                f"Status: {domain.name} research active. Conducted experiment measuring {self.rng.choice(metrics)} during {milestone}",
                f"Current research status: {domain.name} - Replicated {self.rng.choice(concepts)} detection with {self.rng.randint(85, 99)}% {self.rng.choice(metrics)} accuracy",
                f"Research status update: New {self.rng.choice(phenomena)} observation protocol for {domain.name}",
                f"{domain.name} status: Ongoing experiments. Tested {self.rng.choice(particles)} interactions in {self.rng.choice(concepts)} regime"
            ],
            "publication": [
                f"Status: {domain.name} research published. Published findings on {domain.name}: {milestone}",
                f"Research status: Active publication phase. Preprint released: {self.rng.choice(concepts)} advances in {self.rng.choice(phenomena)}",
                f"Status update: Review article on {domain.name} progress through {year}, focusing on {self.rng.choice(metrics)}"
            ],
            "meeting": [
                f"Status: {domain.name} under active discussion. Conference presentation on {domain.name} {self.rng.choice(concepts)}",
                f"Current status: Team discussion ongoing about {self.rng.choice(phenomena)} implications for {domain.name}",
                f"Research status: Collaborative. Workshop on {self.rng.choice(particles)} applications in {domain.name}"
            ],
            "discovery": [
                f"Status: {domain.name} breakthrough achieved. Unexpected {self.rng.choice(phenomena)} behavior observed: {self.rng.choice(metrics)} anomaly",
                f"Research status: Major discovery. Novel {self.rng.choice(concepts)} configuration discovered in {domain.name}",
                f"Status update: Breakthrough in {domain.name}. {self.rng.choice(particles)} exhibits {self.rng.choice(phenomena)} under new conditions"
            ],
            "revision": [
                f"Status: {domain.name} model revised. Updated model: {self.rng.choice(concepts)} assumptions corrected",
                f"Current status: Under revision. Revised {self.rng.choice(metrics)} calibration based on {self.rng.choice(phenomena)} data",
                f"Research status: Reconciling data. Reconciled conflicting {self.rng.choice(particles)} measurements in {domain.name}"
            ],
            "collaboration": [
                f"Status: {domain.name} collaborative phase. Partnership formed for {self.rng.choice(concepts)} research",
                f"Research status: Multi-institutional. Multi-site {self.rng.choice(phenomena)} study initiated for {domain.name}",
                f"Current status: Industry partnership. Collaboration on {self.rng.choice(metrics)} optimization in {domain.name}"
            ],
            "funding": [
                f"Status: {domain.name} funded for development. Grant awarded for {self.rng.choice(concepts)} development",
                f"Research status: Well-funded. Investment in {self.rng.choice(phenomena)} infrastructure for {domain.name}",
                f"Status update: Public funding secured for {self.rng.choice(particles)} research program in {domain.name}"
            ]
        }
        
        base = self.rng.choice(templates.get(event_type, templates["experiment"]))
        
        # Add perspective-specific framing
        perspective_frames = {
            Perspective.OPTIMIST: f" {self.rng.choice(metrics)} exceeded projections.",
            Perspective.SKEPTIC: f" Significant {self.rng.choice(concepts)} challenges remain.",
            Perspective.METHODOLOGIST: f" Rigorous {self.rng.choice(metrics)} protocols followed.",
            Perspective.THEORIST: f" Aligns with {self.rng.choice(phenomena)} predictions.",
            Perspective.EMPIRICIST: f" {self.rng.choice(metrics)}-driven conclusions drawn."
        }
        
        return base + perspective_frames.get(perspective, "")
    
    def _generate_consensus_claim(
        self, 
        domain: AlternateResearchDomain, 
        year: int
    ) -> Claim:
        """Generate a claim all perspectives agree on"""
        milestone = domain.milestones.get(year, "progress")
        content = f"{domain.name}: {milestone}"
        claim_id = self._generate_claim_id(f"{content}_{year}")
        
        # All perspectives agree with high confidence
        perspective_views = {p.value: content for p in Perspective}
        confidence = {p.value: 0.85 + self.rng.random() * 0.15 for p in Perspective}
        
        return Claim(
            id=claim_id,
            content=content,
            claim_type=ClaimType.CONSENSUS,
            topic=domain.name,
            timestamp=self._generate_timestamp(year),
            perspective_views=perspective_views,
            confidence_by_perspective=confidence
        )
    
    def _generate_disputed_claim(
        self,
        domain: AlternateResearchDomain,
        year: int
    ) -> Claim:
        """Generate a claim where perspectives disagree"""
        concepts = list(domain.key_concepts.keys())
        metrics = list(domain.metrics.keys())
        
        base_claim = f"{domain.name} will achieve {self.rng.choice(concepts)} milestone by {year + self.rng.randint(1, 5)}"
        claim_id = self._generate_claim_id(f"{base_claim}_{year}")
        
        # Each perspective has different view
        perspective_modifiers = {
            Perspective.OPTIMIST: (f"ahead of schedule, {self.rng.choice(metrics)} trending positive", 0.9),
            Perspective.SKEPTIC: (f"delayed by {self.rng.choice(concepts)} limitations", 0.75),
            Perspective.METHODOLOGIST: (f"contingent on {self.rng.choice(metrics)} standardization", 0.7),
            Perspective.THEORIST: (f"depends on {self.rng.choice(concepts)} theoretical advances", 0.8),
            Perspective.EMPIRICIST: (f"based on current {self.rng.choice(metrics)} extrapolation", 0.85)
        }
        
        perspective_views = {}
        confidence = {}
        
        for perspective, (modifier, conf) in perspective_modifiers.items():
            view = f"{base_claim} [{perspective.value}: {modifier}]"
            perspective_views[perspective.value] = view
            confidence[perspective.value] = conf + self.rng.uniform(-0.1, 0.1)
        
        return Claim(
            id=claim_id,
            content=base_claim,
            claim_type=ClaimType.DISPUTED,
            topic=domain.name,
            timestamp=self._generate_timestamp(year),
            perspective_views=perspective_views,
            confidence_by_perspective=confidence
        )
    
    def _generate_noise_claim(self, domain: AlternateResearchDomain, year: int) -> Claim:
        """Generate low-quality claim that should be rejected"""
        concepts = list(domain.key_concepts.keys())
        
        noise_templates = [
            f"Unverified rumor about {domain.name} {self.rng.choice(concepts)} breakthrough",
            f"Retracted {domain.name} finding",
            f"Preliminary {self.rng.choice(concepts)} result, not peer-reviewed",
            f"Conflicting {domain.name} data, source unclear"
        ]
        
        content = self.rng.choice(noise_templates)
        claim_id = self._generate_claim_id(f"noise_{content}_{year}")
        
        # All perspectives have low confidence
        confidence = {p.value: self.rng.uniform(0.1, 0.4) for p in Perspective}
        
        return Claim(
            id=claim_id,
            content=content,
            claim_type=ClaimType.NOISE,
            topic=domain.name,
            timestamp=self._generate_timestamp(year),
            perspective_views={p.value: content for p in Perspective},
            confidence_by_perspective=confidence
        )
    
    def _find_causal_parents(
        self,
        domain: AlternateResearchDomain,
        year: int,
        event_type: str
    ) -> List[str]:
        """Find events this event causally depends on"""
        candidates = [
            e for e in self.events
            if e.topic == domain.name 
            and int(e.timestamp[:4]) < year
        ]
        
        if not candidates:
            return []
        
        dependency_map = {
            "experiment": ["funding", "collaboration"],
            "publication": ["experiment", "discovery"],
            "meeting": ["publication", "experiment"],
            "discovery": ["experiment", "publication"],
            "revision": ["discovery", "publication"],
            "collaboration": ["meeting", "funding"],
            "funding": ["publication", "discovery"]
        }
        
        preferred_parents = dependency_map.get(event_type, [])
        typed_candidates = [c for c in candidates if c.event_type in preferred_parents]
        if typed_candidates:
            candidates = typed_candidates
        
        num_parents = self.rng.randint(1, min(3, len(candidates)))
        parents = self.rng.sample(candidates, min(num_parents, len(candidates)))
        return [p.id for p in parents]
    
    def generate_events(self) -> List[Event]:
        """Generate all events across all domains and years"""
        print(f"Generating events for {len(self.domains)} domains, "
              f"{self.end_year - self.start_year + 1} years...")
        
        event_count = 0
        
        for domain in self.domains:
            print(f"  Processing {domain.name}...")
            
            for year in range(self.start_year, self.end_year + 1):
                # Generate consensus claims
                claim = self._generate_consensus_claim(domain, year)
                self.claims[claim.id] = claim
                
                # Generate disputed claims (30% chance per year)
                if self.rng.random() < 0.3:
                    claim = self._generate_disputed_claim(domain, year)
                    self.claims[claim.id] = claim
                
                # Generate noise claims (5% chance)
                if self.rng.random() < 0.05:
                    claim = self._generate_noise_claim(domain, year)
                    self.claims[claim.id] = claim
                
                # Generate events
                for _ in range(self.events_per_topic_per_year):
                    event_type = self._select_event_type()
                    perspectives = self.rng.sample(
                        list(Perspective), 
                        self.perspectives_per_event
                    )
                    
                    for perspective in perspectives:
                        event_id = str(uuid.uuid4())[:8]
                        timestamp = self._generate_timestamp(year)
                        content = self._generate_event_content(
                            domain, year, event_type, perspective
                        )
                        
                        year_claims = [
                            c.id for c in self.claims.values()
                            if c.topic == domain.name 
                            and c.timestamp[:4] == str(year)
                        ]
                        event_claims = self.rng.sample(
                            year_claims, 
                            min(self.rng.randint(1, 3), len(year_claims))
                        ) if year_claims else []
                        
                        parents = self._find_causal_parents(domain, year, event_type)
                        
                        event = Event(
                            id=event_id,
                            timestamp=timestamp,
                            topic=domain.name,
                            event_type=event_type,
                            content=content,
                            claims=event_claims,
                            causal_parents=parents,
                            agent_snapshot_id=f"agent_{year}_{event_id[:4]}",
                            perspective=perspective
                        )
                        
                        self.events.append(event)
                        event_count += 1
                        
                        for parent in parents:
                            self.causal_graph[parent].append(event_id)
        
        print(f"  Generated {event_count} events, {len(self.claims)} claims")
        return self.events
    
    def generate_queries(self) -> List[Query]:
        """Generate diverse benchmark queries with ground truth"""
        print("Generating benchmark queries...")
        
        # Temporal queries
        for domain in self.domains:
            for year in range(self.start_year, self.end_year + 1):
                year_events = [
                    e for e in self.events 
                    if e.topic == domain.name and e.timestamp[:4] == str(year)
                ]
                
                if year_events:
                    self.queries.append(Query(
                        id=f"temporal_{domain.name}_{year}".replace(" ", "_"),
                        question=f"What was the status of {domain.name} research in {year}?",
                        query_type="temporal_recall",
                        timestamp_context=f"{year}-12-31",
                        expected_consensus=[],
                        expected_disputed=[],
                        expected_sources=[e.id for e in year_events[:5]],
                        difficulty="easy"
                    ))
        
        # Consensus queries
        for domain in self.domains:
            consensus_claims = [
                c for c in self.claims.values()
                if c.topic == domain.name and c.claim_type == ClaimType.CONSENSUS
            ]
            disputed_claims = [
                c for c in self.claims.values()
                if c.topic == domain.name and c.claim_type == ClaimType.DISPUTED
            ]
            
            if consensus_claims or disputed_claims:
                self.queries.append(Query(
                    id=f"consensus_{domain.name}".replace(" ", "_"),
                    question=f"What findings about {domain.name} are agreed upon by all perspectives, and what remains disputed?",
                    query_type="consensus_detection",
                    timestamp_context=f"{self.end_year}-12-31",
                    expected_consensus=[c.id for c in consensus_claims],
                    expected_disputed=[
                        {"claim_id": c.id, "perspective_views": c.perspective_views}
                        for c in disputed_claims
                    ],
                    expected_sources=[],
                    difficulty="medium"
                ))
        
        # Perspective queries
        for domain in self.domains:
            for p1, p2 in itertools.combinations(Perspective, 2):
                p1_events = [e for e in self.events if e.topic == domain.name and e.perspective == p1]
                p2_events = [e for e in self.events if e.topic == domain.name and e.perspective == p2]
                
                if p1_events and p2_events:
                    self.queries.append(Query(
                        id=f"perspective_{domain.name}_{p1.value}_{p2.value}".replace(" ", "_"),
                        question=f"How do {p1.value} and {p2.value} perspectives differ on {domain.name}?",
                        query_type="perspective_divergence",
                        timestamp_context=f"{self.end_year}-12-31",
                        expected_consensus=[],
                        expected_disputed=[],
                        expected_sources=[e.id for e in (p1_events[:3] + p2_events[:3])],
                        difficulty="hard"
                    ))
        
        print(f"  Generated {len(self.queries)} queries")
        return self.queries
    
    def generate_dataset(self) -> Dict:
        """Generate complete benchmark dataset"""
        self.generate_events()
        self.generate_queries()
        
        return {
            "metadata": {
                "generator": "SyntheticHistoryGeneratorV2",
                "version": "2.0-phonotactic",
                "description": "Uses phonotactic nouns and alternate physics to eliminate prior knowledge confounds",
                "start_year": self.start_year,
                "end_year": self.end_year,
                "num_events": len(self.events),
                "num_claims": len(self.claims),
                "num_queries": len(self.queries),
                "domains": [d.name for d in self.domains],
                "perspectives": [p.value for p in Perspective]
            },
            "glossary": self.glossary,  # Essential for understanding generated terms
            "events": [e.to_dict() for e in self.events],
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
            "queries": [q.to_dict() for q in self.queries],
            "causal_graph": dict(self.causal_graph),
            "ground_truth": {
                "consensus_claims": [
                    c.id for c in self.claims.values() 
                    if c.claim_type == ClaimType.CONSENSUS
                ],
                "disputed_claims": [
                    c.id for c in self.claims.values()
                    if c.claim_type == ClaimType.DISPUTED
                ],
                "noise_claims": [
                    c.id for c in self.claims.values()
                    if c.claim_type == ClaimType.NOISE
                ]
            }
        }
    
    def save_to_file(self, filename: str = "dpr_rc_benchmark_v2.json"):
        """Save dataset to JSON file"""
        data = self.generate_dataset()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nDataset saved to {filename}")
        print(f"  Events: {data['metadata']['num_events']}")
        print(f"  Claims: {data['metadata']['num_claims']}")
        print(f"  Queries: {data['metadata']['num_queries']}")
        print(f"  Domains: {data['metadata']['domains']}")
        
        return data


def main():
    print("=" * 70)
    print("SYNTHETIC HISTORY GENERATOR V2")
    print("With Phonotactic Nouns and Alternate Universe Physics")
    print("=" * 70)
    
    # Generate medium dataset
    gen = SyntheticHistoryGeneratorV2(
        events_per_topic_per_year=50,
        perspectives_per_event=3,
        num_domains=4,
        seed=42
    )
    data = gen.save_to_file("dpr_rc_benchmark_phonotactic.json")
    
    # Show sample terminology
    print("\n" + "=" * 70)
    print("SAMPLE GENERATED TERMINOLOGY")
    print("=" * 70)
    
    print("\nDomains:")
    for domain in data['metadata']['domains']:
        print(f"  - {domain}")
    
    print("\nSample Particles:")
    for name, props in list(data['glossary']['physics']['particles'].items())[:3]:
        print(f"  - {name}: {props['type']}, charge={props['charge']}, spin={props['spin']}")
    
    print("\nSample Phenomena:")
    for name, props in list(data['glossary']['physics']['phenomena'].items())[:3]:
        print(f"  - {name}: {props['type']}, particles={props['particles_involved'][:2]}")
    
    print("\nSample Event Content:")
    for event in data['events'][:3]:
        print(f"  [{event['timestamp'][:4]}] {event['content'][:80]}...")
    
    print("\n" + "=" * 70)
    print("This terminology has ZERO overlap with real-world knowledge!")
    print("=" * 70)


if __name__ == "__main__":
    main()
