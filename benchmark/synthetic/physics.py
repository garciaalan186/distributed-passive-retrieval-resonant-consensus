"""
Alternate Universe Physics Generator

Creates a self-consistent alternate universe physics system with
fundamental particles, constants, phenomena, and units.
"""

import random
from typing import Dict

from benchmark.synthetic.phonotactic import PhonotacticGenerator


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

        self.particles = self._generate_particles()
        self.constants = self._generate_constants()
        self.phenomena = self._generate_phenomena()
        self.units = self._generate_units()

    def _generate_particles(self) -> Dict[str, Dict]:
        """Generate alternate universe fundamental particles"""
        particles = {}

        particle_types = [
            ('fermion', 6),
            ('boson', 4),
            ('lepton', 4),
            ('exotic', 3)
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
            ('speed_limit', 'velocity', 1e6, 1e10),
            ('coupling', 'dimensionless', 0.001, 0.1),
            ('quantum', 'action', 1e-35, 1e-32),
            ('field_strength', 'force', 1e-12, 1e-8),
            ('entropy_base', 'temperature', 1e-24, 1e-20)
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
            'field_effect',
            'quantum_effect',
            'thermodynamic',
            'relativistic',
            'emergent'
        ]

        for ptype in phenomenon_types:
            for i in range(3):
                name = self.phono.generate_word('process')

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
