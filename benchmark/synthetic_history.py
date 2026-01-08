"""
Synthetic History Generator - Backward Compatibility Module

This module re-exports all classes from the new benchmark.synthetic package
for backward compatibility. New code should import from benchmark.synthetic.

Example:
    # Old (deprecated but still works):
    from benchmark.synthetic_history import SyntheticHistoryGeneratorV2

    # New (preferred):
    from benchmark.synthetic import SyntheticHistoryGeneratorV2
"""

from benchmark.synthetic import (
    Perspective,
    ClaimType,
    Claim,
    Event,
    Query,
    REAL_WORLD_FORBIDDEN_TERMS,
    PhonotacticGenerator,
    AlternatePhysics,
    AlternateResearchDomain,
    SyntheticHistoryGeneratorV2,
)

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


def main():
    """Run main generator (for backward compatibility with CLI usage)."""
    print("=" * 70)
    print("SYNTHETIC HISTORY GENERATOR V2")
    print("With Phonotactic Nouns and Alternate Universe Physics")
    print("=" * 70)

    gen = SyntheticHistoryGeneratorV2(
        events_per_topic_per_year=50,
        perspectives_per_event=3,
        num_domains=4,
        seed=42
    )
    data = gen.save_to_file("dpr_rc_benchmark_phonotactic.json")

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
