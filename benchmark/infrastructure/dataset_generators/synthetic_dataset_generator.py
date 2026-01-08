"""
Synthetic Dataset Generator

Adapter that wraps SyntheticHistoryGeneratorV2 to implement IDatasetGenerator interface.

This adapter provides a clean interface for use cases while maintaining backward
compatibility with the existing synthetic history generation logic.
"""

from typing import List, Dict, Any, Optional
from benchmark.domain.interfaces import IDatasetGenerator, BenchmarkDataset
from benchmark.synthetic import SyntheticHistoryGeneratorV2


class SyntheticDatasetGenerator(IDatasetGenerator):
    """
    Generates synthetic benchmark datasets using phonotactic terms.

    This adapter wraps SyntheticHistoryGeneratorV2 to implement the clean
    IDatasetGenerator interface for use in RunBenchmarkUseCase.

    Design Principles:
    1. **Reproducibility**: Same scale + seed always produces same dataset
    2. **No Data Leakage**: Uses phonotactic terms to eliminate prior knowledge
    3. **Controlled Complexity**: Scale levels systematically vary difficulty
    4. **Backward Compatibility**: Uses existing SyntheticHistoryGeneratorV2

    Scale Configuration:
    - small: 10 events/topic/year, 2 domains
    - medium: 25 events/topic/year, 3 domains
    - large: 50 events/topic/year, 4 domains
    - stress: 100 events/topic/year, 5 domains
    """

    # Scale configuration matching ResearchBenchmarkSuite
    SCALE_CONFIGS = {
        "small": {
            "events_per_topic_per_year": 10,
            "num_domains": 2,
            "perspectives_per_event": 3,
        },
        "medium": {
            "events_per_topic_per_year": 25,
            "num_domains": 3,
            "perspectives_per_event": 3,
        },
        "large": {
            "events_per_topic_per_year": 50,
            "num_domains": 4,
            "perspectives_per_event": 3,
        },
        "stress": {
            "events_per_topic_per_year": 100,
            "num_domains": 5,
            "perspectives_per_event": 3,
        },
    }

    def generate(
        self,
        scale: str,
        seed: Optional[int] = None
    ) -> BenchmarkDataset:
        """
        Generate a synthetic benchmark dataset.

        Args:
            scale: Scale level ('small', 'medium', 'large', 'stress')
            seed: Optional random seed for reproducibility

        Returns:
            BenchmarkDataset with queries and ground truth

        Raises:
            ValueError: If scale is invalid

        Implementation Notes:
            - Uses SyntheticHistoryGeneratorV2 internally
            - Converts its output format to BenchmarkDataset
            - Preserves all ground truth data (consensus, disputed, sources)
            - Includes glossary for hallucination detection
        """
        if scale not in self.SCALE_CONFIGS:
            raise ValueError(
                f"Invalid scale '{scale}'. Valid options: {list(self.SCALE_CONFIGS.keys())}"
            )

        config = self.SCALE_CONFIGS[scale]

        # Create generator with scale configuration
        generator = SyntheticHistoryGeneratorV2(
            events_per_topic_per_year=config["events_per_topic_per_year"],
            perspectives_per_event=config["perspectives_per_event"],
            num_domains=config["num_domains"],
            seed=seed
        )

        # Generate dataset using existing logic
        dataset_dict = generator.generate_dataset()

        # Convert to BenchmarkDataset
        return BenchmarkDataset(
            scale=scale,
            queries=dataset_dict["queries"],
            glossary=generator.glossary,
            metadata={
                "num_events": len(dataset_dict["events"]),
                "num_queries": len(dataset_dict["queries"]),
                "generation_config": config,
                "seed": seed,
                "generator_version": "SyntheticHistoryGeneratorV2"
            }
        )

    def get_scale_config(self, scale: str) -> Dict[str, Any]:
        """
        Get configuration parameters for a scale level.

        Args:
            scale: Scale level name

        Returns:
            Dictionary with scale parameters

        Raises:
            ValueError: If scale is invalid
        """
        if scale not in self.SCALE_CONFIGS:
            raise ValueError(
                f"Invalid scale '{scale}'. Valid options: {list(self.SCALE_CONFIGS.keys())}"
            )

        config = self.SCALE_CONFIGS[scale].copy()

        # Add estimated query count (approximate based on config)
        # SyntheticHistoryGeneratorV2 generates queries based on events and domains
        # This is an estimate; actual count determined by generation logic
        events_estimate = (
            config["events_per_topic_per_year"] *
            config["num_domains"] *
            3  # Approximate topics per domain
        )
        config["expected_query_count"] = int(events_estimate * 0.3)  # ~30% of events become queries

        return config

    def validate_dataset(self, dataset: BenchmarkDataset) -> List[str]:
        """
        Validate a dataset for quality and completeness.

        Args:
            dataset: The dataset to validate

        Returns:
            List of validation warnings/errors (empty if valid)

        Validation Checks:
            - All queries have ground truth
            - No duplicate query IDs
            - Glossary is complete
            - Metadata is present
            - Scale matches configuration
        """
        warnings = []

        # Check scale
        if dataset.scale not in self.SCALE_CONFIGS:
            warnings.append(f"Unknown scale: {dataset.scale}")

        # Check queries
        if not dataset.queries:
            warnings.append("Dataset has no queries")
        else:
            # Check for ground truth
            queries_without_ground_truth = []
            for i, query in enumerate(dataset.queries):
                if "expected_consensus" not in query and "expected_disputed" not in query:
                    queries_without_ground_truth.append(i)

            if queries_without_ground_truth:
                warnings.append(
                    f"{len(queries_without_ground_truth)} queries missing ground truth "
                    f"(indices: {queries_without_ground_truth[:5]}...)"
                )

            # Check for duplicates
            query_texts = [q.get("question", "") for q in dataset.queries]
            if len(query_texts) != len(set(query_texts)):
                warnings.append("Dataset contains duplicate queries")

        # Check glossary
        if not dataset.glossary:
            warnings.append("Dataset has no glossary (hallucination detection will fail)")
        else:
            # Check glossary structure
            if "physics" not in dataset.glossary and "domains" not in dataset.glossary:
                warnings.append("Glossary missing expected sections (physics, domains)")

        # Check metadata
        if not dataset.metadata:
            warnings.append("Dataset has no metadata")
        else:
            expected_metadata_keys = ["num_events", "num_queries", "generation_config"]
            missing_keys = [
                key for key in expected_metadata_keys
                if key not in dataset.metadata
            ]
            if missing_keys:
                warnings.append(f"Metadata missing keys: {missing_keys}")

        return warnings
