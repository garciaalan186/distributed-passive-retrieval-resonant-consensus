"""
Dataset Generator Interface

This interface defines how benchmark datasets are generated.
Generators MUST produce reproducible datasets for scientific validity.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkDataset:
    """
    A complete benchmark dataset with queries and ground truth.

    Attributes:
        scale: Scale level ('small', 'medium', 'large', 'stress')
        queries: List of query dictionaries with ground truth
        glossary: Domain-specific glossary (for hallucination detection)
        metadata: Additional dataset metadata (generation params, etc.)
    """
    scale: str
    queries: List[Dict[str, Any]]
    glossary: Dict[str, Any]
    metadata: Dict[str, Any]


class IDatasetGenerator(ABC):
    """
    Interface for generating benchmark datasets.

    CRITICAL REQUIREMENTS FOR SCIENTIFIC VALIDITY:

    1. **Reproducibility**: Given the same parameters (scale, seed),
       generators MUST produce identical datasets. This ensures:
       - Comparable results across runs
       - Ability to reproduce published results
       - Valid statistical analysis

    2. **No Data Leakage**: Generated queries must not leak information
       about the system being tested or its implementation details.

    3. **Controlled Complexity**: Different scale levels should systematically
       vary complexity in measurable ways (number of events, perspectives, etc.)

    4. **Ground Truth Quality**: Ground truth must be unambiguous and verifiable.

    Implementations:
    - SyntheticHistoryGenerator: Generates synthetic historical events
    - RealDatasetLoader: Loads real-world benchmark datasets
    - MockDatasetGenerator: For testing
    """

    @abstractmethod
    def generate(
        self,
        scale: str,
        seed: Optional[int] = None
    ) -> BenchmarkDataset:
        """
        Generate a benchmark dataset for the specified scale.

        Args:
            scale: Scale level ('small', 'medium', 'large', 'stress')
            seed: Optional random seed for reproducibility

        Returns:
            BenchmarkDataset with queries and ground truth

        Raises:
            ValueError: If scale is invalid

        Note:
            MUST be deterministic when seed is provided. Same scale + seed
            must always produce the same dataset.
        """
        pass

    @abstractmethod
    def get_scale_config(self, scale: str) -> Dict[str, Any]:
        """
        Get configuration parameters for a scale level.

        Args:
            scale: Scale level name

        Returns:
            Dictionary with scale parameters:
            - 'events_per_topic_per_year': int
            - 'num_domains': int
            - 'expected_query_count': int
            - etc.

        Note:
            This enables benchmark reports to document the exact
            parameters used for dataset generation.
        """
        pass

    @abstractmethod
    def validate_dataset(self, dataset: BenchmarkDataset) -> List[str]:
        """
        Validate a dataset for quality and completeness.

        Args:
            dataset: The dataset to validate

        Returns:
            List of validation warnings/errors (empty if valid)

        Note:
            This method checks:
            - All queries have ground truth
            - No duplicate queries
            - Glossary is complete
            - Metadata is present
        """
        pass
