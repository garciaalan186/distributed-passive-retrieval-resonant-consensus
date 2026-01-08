"""
Alternate Research Domain Generator

Creates research domains in the alternate universe with
self-consistent terminology, milestones, and metrics.
"""

import random
from typing import Dict

from benchmark.synthetic.phonotactic import PhonotacticGenerator
from benchmark.synthetic.physics import AlternatePhysics


class AlternateResearchDomain:
    """
    A research domain in the alternate universe with self-consistent terminology.
    """

    def __init__(self, name: str, physics: AlternatePhysics, seed: int = 42):
        self.name = name
        self.physics = physics
        self.rng = random.Random(seed)
        self.phono = PhonotacticGenerator(seed + hash(name) % 10000)

        self.key_concepts = self._generate_concepts()
        self.milestones = self._generate_milestones()
        self.metrics = self._generate_metrics()

    def _generate_concepts(self) -> Dict[str, Dict]:
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
                'yearly_improvement': self.rng.uniform(1.2, 2.0)
            }

        return metrics
