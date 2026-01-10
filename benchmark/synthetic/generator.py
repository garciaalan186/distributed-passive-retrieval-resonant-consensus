"""
Synthetic History Generator V2

Main orchestrator for generating benchmark datasets with phonotactic nouns
and alternate universe physics to eliminate prior knowledge confounds.
"""

import json
import random
import hashlib
import uuid
import itertools
from typing import List, Dict
from collections import defaultdict

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
from benchmark.config import get_config, get_all_scales


def _get_scale_configs() -> Dict:
    """Load scale configurations from YAML config files."""
    return get_all_scales()


# For backward compatibility
SCALE_CONFIGS = _get_scale_configs()


class SyntheticHistoryGeneratorV2:
    """
    Generates large-scale synthetic research history using phonotactic nouns
    and alternate universe physics to eliminate prior knowledge confounds.
    """

    def __init__(
        self,
        start_year: int = None,
        end_year: int = None,
        events_per_topic_per_year: int = 100,
        perspectives_per_event: int = 3,
        num_domains: int = 4,
        seed: int = None
    ):
        # Load defaults from config
        synthetic_config = get_config().get('synthetic', {})
        self.start_year = start_year if start_year is not None else synthetic_config.get('start_year', 2015)
        self.end_year = end_year if end_year is not None else synthetic_config.get('end_year', 2025)
        seed = seed if seed is not None else synthetic_config.get('seed', 42)
        self.events_per_topic_per_year = events_per_topic_per_year
        self.perspectives_per_event = perspectives_per_event
        self.rng = random.Random(seed)

        print("Generating alternate universe physics...")
        self.physics = AlternatePhysics(seed)

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

        self.event_types = [
            ("experiment", 0.3),
            ("publication", 0.25),
            ("meeting", 0.15),
            ("discovery", 0.1),
            ("revision", 0.1),
            ("collaboration", 0.05),
            ("funding", 0.05)
        ]

        self.events: List[Event] = []
        self.claims: Dict[str, Claim] = {}
        self.queries: List[Query] = []
        self.causal_graph: Dict[str, List[str]] = defaultdict(list)

        self.glossary = self._build_glossary()

    def generate(self, scale: str = "medium") -> Dict:
        """Generate dataset for a specific scale level."""
        config = SCALE_CONFIGS.get(scale, SCALE_CONFIGS["medium"])

        self.events_per_topic_per_year = config["events_per_topic_per_year"]
        self.perspectives_per_event = config["perspectives_per_event"]

        return self.generate_dataset()

    def get_glossary(self) -> Dict:
        """Return the glossary of generated terms."""
        return self.glossary

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

    def _get_all_valid_terms(self) -> set:
        """Get all valid synthetic terms from physics and domains."""
        terms = set()

        terms.update(self.physics.particles.keys())
        terms.update(self.physics.constants.keys())
        terms.update(self.physics.phenomena.keys())
        terms.update(self.physics.units.values())

        for domain in self.domains:
            terms.add(domain.name)
            terms.update(domain.name.split())
            terms.update(domain.key_concepts.keys())
            terms.update(domain.metrics.keys())

        return terms

    def _extract_terms_from_events(self, event_ids: List[str]) -> List[str]:
        """Extract synthetic terms from event content for validation."""
        valid_terms = self._get_all_valid_terms()
        found_terms = set()

        event_map = {e.id: e for e in self.events}

        for event_id in event_ids:
            event = event_map.get(event_id)
            if event:
                for word in event.content.split():
                    clean = word.strip('.,!?;:()"\'-')
                    if clean in valid_terms:
                        found_terms.add(clean)

        return list(found_terms)

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
        content = f"Research status for {year}: {domain.name} - {milestone}"
        claim_id = self._generate_claim_id(f"{content}_{year}")

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
        target_year = year + self.rng.randint(1, 5)
        base_claim = f"Status projection: {domain.name} will achieve {self.rng.choice(concepts)} milestone by {target_year}"
        claim_id = self._generate_claim_id(f"{base_claim}_{year}")

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
            f"Status uncertain: Unverified rumor about {domain.name} {self.rng.choice(concepts)} breakthrough",
            f"Status: Retracted {domain.name} finding",
            f"Status pending review: Preliminary {self.rng.choice(concepts)} result, not peer-reviewed",
            f"Status unclear: Conflicting {domain.name} data, source uncertain"
        ]

        content = self.rng.choice(noise_templates)
        claim_id = self._generate_claim_id(f"noise_{content}_{year}")

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
                claim = self._generate_consensus_claim(domain, year)
                self.claims[claim.id] = claim

                if self.rng.random() < 0.3:
                    claim = self._generate_disputed_claim(domain, year)
                    self.claims[claim.id] = claim

                if self.rng.random() < 0.05:
                    claim = self._generate_noise_claim(domain, year)
                    self.claims[claim.id] = claim

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

        for domain in self.domains:
            for year in range(self.start_year, self.end_year + 1):
                year_events = [
                    e for e in self.events
                    if e.topic == domain.name and e.timestamp[:4] == str(year)
                ]

                if year_events:
                    event_ids = [e.id for e in year_events[:5]]
                    self.queries.append(Query(
                        id=f"temporal_{domain.name}_{year}".replace(" ", "_"),
                        question=f"What was the status of {domain.name} research in {year}?",
                        query_type="temporal_recall",
                        timestamp_context=f"{year}-12-31",
                        expected_consensus=[],
                        expected_disputed=[],
                        expected_sources=event_ids,
                        difficulty="easy",
                        required_terms=self._extract_terms_from_events(event_ids),
                        forbidden_terms=list(REAL_WORLD_FORBIDDEN_TERMS)
                    ))

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
                domain_events = [e for e in self.events if e.topic == domain.name]
                domain_event_ids = [e.id for e in domain_events[:10]]
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
                    expected_sources=domain_event_ids,
                    difficulty="medium",
                    required_terms=self._extract_terms_from_events(domain_event_ids),
                    forbidden_terms=list(REAL_WORLD_FORBIDDEN_TERMS)
                ))

        for domain in self.domains:
            for p1, p2 in itertools.combinations(Perspective, 2):
                p1_events = [e for e in self.events if e.topic == domain.name and e.perspective == p1]
                p2_events = [e for e in self.events if e.topic == domain.name and e.perspective == p2]

                if p1_events and p2_events:
                    event_ids = [e.id for e in (p1_events[:3] + p2_events[:3])]
                    self.queries.append(Query(
                        id=f"perspective_{domain.name}_{p1.value}_{p2.value}".replace(" ", "_"),
                        question=f"How do {p1.value} and {p2.value} perspectives differ on {domain.name}?",
                        query_type="perspective_divergence",
                        timestamp_context=f"{self.end_year}-12-31",
                        expected_consensus=[],
                        expected_disputed=[],
                        expected_sources=event_ids,
                        difficulty="hard",
                        required_terms=self._extract_terms_from_events(event_ids),
                        forbidden_terms=list(REAL_WORLD_FORBIDDEN_TERMS)
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
            "glossary": self.glossary,
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
