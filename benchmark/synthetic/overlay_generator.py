"""
Overlay Generator for Belief Revision Events

Generates revision/contradiction events that reference existing claims,
enabling testing of temporal RAG systems' ability to handle belief evolution.

Usage:
    from benchmark.synthetic.overlay_generator import OverlayGenerator

    generator = OverlayGenerator(dataset_path="benchmark_results_local/mini/dataset.json")
    overlay = generator.generate()
    generator.save(output_dir="benchmark_results_local/mini/")
"""

import json
import random
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from benchmark.synthetic.models import (
    RevisionType,
    RevisionEvent,
    RevisionMetadata,
    Query,
    Perspective,
)


# Templates for generating revision event content
REVISION_TEMPLATES = {
    RevisionType.DISPROVEN: [
        "Status update: Previous findings on {concept} have been disproven. "
        "The {year_original} claim regarding {domain} has been invalidated by subsequent {new_evidence} analysis.",

        "Research status: {domain} correction issued. The {year_original} hypothesis "
        "is no longer supported by current evidence. Replication attempts failed to confirm original results.",

        "Critical update for {domain}: Earlier conclusions from {year_original} have been retracted. "
        "New {methodology} protocols reveal fundamental errors in the original {concept} measurements.",
    ],
    RevisionType.REFINED: [
        "Status: {domain} model refined. The {year_original} understanding of {concept} "
        "has been updated with higher precision measurements. Previous estimates revised by {delta}%.",

        "Research refinement: {concept} parameters in {domain} recalibrated. "
        "Original {year_original} values remain directionally correct but magnitude adjusted.",

        "{domain} precision update: The {year_original} {concept} framework now incorporates "
        "improved {methodology} data, narrowing uncertainty bounds significantly.",
    ],
    RevisionType.SUPERSEDED: [
        "Status update: {concept} framework superseded. The {year_original} model in {domain} "
        "has been replaced by the {new_model} paradigm as of {year_revision}.",

        "Paradigm shift in {domain}: The {year_original} {concept} approach is now considered obsolete. "
        "Current research uses the {new_model} framework exclusively.",

        "{domain} methodology transition complete: All work previously based on {year_original} "
        "{concept} models should now reference the {new_model} successor framework.",
    ],
}

# Query templates for testing revision awareness
REVISION_QUERY_TEMPLATES = {
    "current_status": [
        "What is the CURRENT understanding of {concept} in {domain}?",
        "What is the latest status of {concept} research in {domain}?",
        "What do we currently know about {concept} in {domain}?",
    ],
    "temporal_belief": [
        "What was believed about {concept} in {domain} during {year}?",
        "What was the status of {concept} research in {domain} as of {year}?",
        "How was {concept} understood in {domain} in {year}?",
    ],
    "belief_evolution": [
        "How has the understanding of {concept} in {domain} changed over time?",
        "What revisions have been made to {concept} findings in {domain}?",
        "Trace the evolution of {concept} research in {domain}.",
    ],
}


class OverlayGenerator:
    """
    Generates belief revision overlay events for existing datasets.

    This generator scans an existing dataset.json, identifies claims
    eligible for revision, and generates revision events that can be
    ingested into the same ChromaDB shards.
    """

    def __init__(
        self,
        dataset_path: str,
        glossary_path: Optional[str] = None,
        revision_rate: float = 0.05,
        seed: int = 42,
        min_revision_gap: int = 2,
        max_revision_gap: int = 5,
    ):
        """
        Initialize the overlay generator.

        Args:
            dataset_path: Path to existing dataset.json
            glossary_path: Path to glossary.json (auto-detected if None)
            revision_rate: Fraction of claims to revise per domain (~5%)
            seed: Random seed for reproducibility
            min_revision_gap: Minimum years between claim and revision
            max_revision_gap: Maximum years between claim and revision
        """
        self.dataset_path = Path(dataset_path)
        self.glossary_path = glossary_path or self.dataset_path.parent / "glossary.json"
        self.revision_rate = revision_rate
        self.min_revision_gap = min_revision_gap
        self.max_revision_gap = max_revision_gap
        self.rng = random.Random(seed)

        # Load existing data
        self.dataset = self._load_json(self.dataset_path)
        self.glossary = self._load_json(self.glossary_path) if Path(self.glossary_path).exists() else {}

        # Parse dataset structure
        self.events = self.dataset.get("events", [])
        # Claims can be either a dict (keyed by id) or a list
        claims_data = self.dataset.get("claims", {})
        if isinstance(claims_data, dict):
            self.claims = claims_data
        else:
            self.claims = {c["id"]: c for c in claims_data}
        self.domains = self._extract_domains()
        self.year_range = self._extract_year_range()

        # Generated overlay data
        self.revision_events: List[RevisionEvent] = []
        self.revision_queries: List[Query] = []
        self.revision_metadata = RevisionMetadata()

    def _load_json(self, path: Path) -> Dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _extract_domains(self) -> List[str]:
        """Extract unique domains from events."""
        domains = set()
        for event in self.events:
            topic = event.get("topic", "")
            if topic:
                domains.add(topic)
        return sorted(domains)

    def _extract_year_range(self) -> Tuple[int, int]:
        """Extract min/max years from event timestamps."""
        years = []
        for event in self.events:
            ts = event.get("timestamp", "")
            if ts and len(ts) >= 4:
                try:
                    years.append(int(ts[:4]))
                except ValueError:
                    pass
        if years:
            return min(years), max(years)
        return 2015, 2025  # Default

    def _extract_concept_from_content(self, content: str, domain: str) -> str:
        """Extract a key concept from event content for use in templates."""
        # Try to find domain-specific terms from glossary
        if self.glossary:
            domain_terms = self.glossary.get("domains", {}).get(domain, {})
            concepts = domain_terms.get("key_concepts", {})
            if concepts:
                # Return a random concept from the domain
                return self.rng.choice(list(concepts.keys()))

        # Try to extract a concept from content that isn't the domain name
        words = content.split()
        domain_words = set(domain.lower().split())

        # Look for capitalized multi-word phrases that aren't the domain
        for i in range(len(words) - 1):
            if (words[i][0].isupper() and words[i+1][0].isupper() and
                words[i].lower() not in domain_words and
                words[i+1].lower() not in domain_words):
                return f"{words[i]} {words[i+1]}"

        # Look for single capitalized words that aren't the domain
        for word in words:
            if (len(word) > 3 and word[0].isupper() and
                word.lower() not in domain_words and
                word.lower() not in ["status", "research", "current", "update"]):
                return word

        # Last resort: use a generic term based on event type indicators
        if "detection" in content.lower():
            return "detection methods"
        elif "measurement" in content.lower():
            return "measurement protocols"
        elif "framework" in content.lower():
            return "theoretical framework"
        else:
            return "core findings"

    def _get_revision_candidates(self, domain: str) -> List[Dict]:
        """
        Find claims eligible for revision in a domain.

        Eligible claims are:
        - From discovery or publication events
        - Made at least min_revision_gap years before end of dataset
        - Not already noise claims
        """
        candidates = []
        end_year = self.year_range[1]

        for event in self.events:
            if event.get("topic") != domain:
                continue
            if event.get("event_type") not in ["discovery", "publication", "experiment"]:
                continue

            ts = event.get("timestamp", "")
            if not ts or len(ts) < 4:
                continue

            try:
                year = int(ts[:4])
            except ValueError:
                continue

            # Must have enough years remaining for revision
            if year > end_year - self.min_revision_gap:
                continue

            # Check if event has claims
            event_claims = event.get("claims", [])
            if not event_claims:
                continue

            candidates.append({
                "event": event,
                "year": year,
                "claims": event_claims,
            })

        return candidates

    def _generate_revision_content(
        self,
        revision_type: RevisionType,
        domain: str,
        concept: str,
        year_original: int,
        year_revision: int,
    ) -> str:
        """Generate content for a revision event."""
        templates = REVISION_TEMPLATES[revision_type]
        template = self.rng.choice(templates)

        # Generate placeholder values
        new_evidence = self.rng.choice([
            "spectroscopic", "interferometric", "computational",
            "multi-site", "longitudinal", "cross-validation"
        ])
        methodology = self.rng.choice([
            "calibration", "measurement", "detection",
            "verification", "analysis", "synthesis"
        ])
        delta = self.rng.randint(15, 45)
        new_model = f"{concept.split()[0]}-{self.rng.choice(['enhanced', 'revised', 'unified', 'extended'])}"

        return template.format(
            concept=concept,
            domain=domain,
            year_original=year_original,
            year_revision=year_revision,
            new_evidence=new_evidence,
            methodology=methodology,
            delta=delta,
            new_model=new_model,
        )

    def generate(self) -> Dict:
        """
        Generate overlay events for all domains.

        Returns:
            Dict with overlay_events, overlay_queries, and revision_metadata
        """
        print(f"Generating belief revision overlay...")
        print(f"  Domains: {len(self.domains)}")
        print(f"  Revision rate: {self.revision_rate*100:.1f}%")
        print(f"  Year range: {self.year_range[0]}-{self.year_range[1]}")

        for domain in self.domains:
            self._generate_domain_revisions(domain)

        # Build claims_valid_at_year index
        self._build_temporal_validity_index()

        print(f"\nGenerated:")
        print(f"  Revision events: {len(self.revision_events)}")
        print(f"  Revision queries: {len(self.revision_queries)}")
        print(f"  Disproven claims: {len(self.revision_metadata.disproven_claims)}")

        return {
            "overlay_events": [r.to_dict() for r in self.revision_events],
            "overlay_queries": [q.to_dict() for q in self.revision_queries],
            "revision_metadata": self.revision_metadata.to_dict(),
        }

    def _generate_domain_revisions(self, domain: str):
        """Generate revisions for a single domain."""
        candidates = self._get_revision_candidates(domain)
        if not candidates:
            return

        # Select candidates for revision
        num_revisions = max(1, int(len(candidates) * self.revision_rate))
        selected = self.rng.sample(candidates, min(num_revisions, len(candidates)))

        for candidate in selected:
            self._generate_single_revision(domain, candidate)

    def _generate_single_revision(self, domain: str, candidate: Dict):
        """Generate a single revision event."""
        event = candidate["event"]
        original_year = candidate["year"]
        end_year = self.year_range[1]

        # Determine revision year
        max_gap = min(self.max_revision_gap, end_year - original_year)
        if max_gap < self.min_revision_gap:
            return

        revision_year = original_year + self.rng.randint(self.min_revision_gap, max_gap)

        # Select revision type
        revision_type = self.rng.choice([
            RevisionType.DISPROVEN,
            RevisionType.REFINED,
            RevisionType.SUPERSEDED,
        ])

        # Get original claim info
        original_claim_id = candidate["claims"][0] if candidate["claims"] else event["id"]
        original_claim_content = event.get("content", "")

        # Extract concept for templates
        concept = self._extract_concept_from_content(original_claim_content, domain)

        # Generate revision content
        content = self._generate_revision_content(
            revision_type, domain, concept, original_year, revision_year
        )

        # Create revision event
        revision_id = f"rev_{uuid.uuid4().hex[:8]}"
        revision_event = RevisionEvent(
            id=revision_id,
            original_claim_id=original_claim_id,
            original_claim_content=original_claim_content[:200],  # Truncate for storage
            original_year=original_year,
            revision_year=revision_year,
            revision_type=revision_type,
            content=content,
            domain=domain,
        )

        self.revision_events.append(revision_event)

        # Update metadata
        if revision_type == RevisionType.DISPROVEN:
            if original_claim_id not in self.revision_metadata.disproven_claims:
                self.revision_metadata.disproven_claims[original_claim_id] = []
            self.revision_metadata.disproven_claims[original_claim_id].append(revision_id)

        self.revision_metadata.revision_events.append(revision_event)

        # Generate queries for this revision
        self._generate_revision_queries(domain, concept, revision_event)

    def _generate_revision_queries(
        self,
        domain: str,
        concept: str,
        revision_event: RevisionEvent
    ):
        """Generate test queries for a revision event."""
        # Current status query (should NOT return disproven claim)
        if revision_event.revision_type == RevisionType.DISPROVEN:
            template = self.rng.choice(REVISION_QUERY_TEMPLATES["current_status"])
            query = Query(
                id=f"revision_current_{revision_event.id}",
                question=template.format(concept=concept, domain=domain),
                query_type="current_status",
                timestamp_context=f"{self.year_range[1]}-12-31",
                expected_consensus=[],
                expected_disputed=[],
                expected_sources=[revision_event.id],
                difficulty="hard",
                required_terms=[],
                forbidden_terms=[revision_event.original_claim_content[:50]],  # Should NOT appear
            )
            self.revision_queries.append(query)

        # Temporal belief query (should return claim valid at that time)
        template = self.rng.choice(REVISION_QUERY_TEMPLATES["temporal_belief"])
        query = Query(
            id=f"revision_temporal_{revision_event.id}",
            question=template.format(
                concept=concept,
                domain=domain,
                year=revision_event.original_year
            ),
            query_type="temporal_belief",
            timestamp_context=f"{revision_event.original_year}-12-31",
            expected_consensus=[],
            expected_disputed=[],
            expected_sources=[revision_event.original_claim_id],
            difficulty="medium",
            required_terms=[],
            forbidden_terms=[],
        )
        self.revision_queries.append(query)

        # Belief evolution query
        template = self.rng.choice(REVISION_QUERY_TEMPLATES["belief_evolution"])
        query = Query(
            id=f"revision_evolution_{revision_event.id}",
            question=template.format(concept=concept, domain=domain),
            query_type="belief_evolution",
            timestamp_context=f"{self.year_range[1]}-12-31",
            expected_consensus=[],
            expected_disputed=[],
            expected_sources=[revision_event.original_claim_id, revision_event.id],
            difficulty="hard",
            required_terms=["revised", "updated", "corrected", "superseded", "disproven"],
            forbidden_terms=[],
            validation_pattern="any_required",  # Any one of required terms is sufficient
        )
        self.revision_queries.append(query)

    def _build_temporal_validity_index(self):
        """Build index of which claims are valid at each year."""
        # Start with all claims valid
        all_claim_ids = set()
        for event in self.events:
            for claim_id in event.get("claims", []):
                all_claim_ids.add(claim_id)

        # For each year, compute valid claims
        for year in range(self.year_range[0], self.year_range[1] + 1):
            valid_at_year = set(all_claim_ids)

            # Remove claims that were disproven before this year
            for revision_event in self.revision_events:
                if revision_event.revision_type == RevisionType.DISPROVEN:
                    if revision_event.revision_year <= year:
                        valid_at_year.discard(revision_event.original_claim_id)

            self.revision_metadata.claims_valid_at_year[year] = list(valid_at_year)

    def save(self, output_dir: str):
        """Save overlay data to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save overlay events
        overlay_events_path = output_path / "overlay_events.json"
        with open(overlay_events_path, 'w') as f:
            json.dump([r.to_dict() for r in self.revision_events], f, indent=2)
        print(f"Saved overlay events to {overlay_events_path}")

        # Save overlay queries
        overlay_queries_path = output_path / "overlay_queries.json"
        with open(overlay_queries_path, 'w') as f:
            json.dump([q.to_dict() for q in self.revision_queries], f, indent=2)
        print(f"Saved overlay queries to {overlay_queries_path}")

        # Save revision metadata
        metadata_path = output_path / "overlay_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.revision_metadata.to_dict(), f, indent=2)
        print(f"Saved revision metadata to {metadata_path}")

    def get_events_for_ingestion(self) -> List[Dict]:
        """
        Get revision events formatted for ChromaDB ingestion.

        Returns list of event dicts compatible with existing ingestion pipeline.
        """
        return [r.to_event().to_dict() for r in self.revision_events]
