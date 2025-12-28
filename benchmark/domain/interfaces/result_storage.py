"""
Result Storage Interface

This interface defines how benchmark results are persisted and retrieved.
Implementations handle different storage backends (local, GCS, S3, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pathlib import Path


class IResultStorage(ABC):
    """
    Interface for storing and retrieving benchmark results.

    This interface abstracts the storage layer, allowing benchmarks to work
    with different storage backends without changing benchmark logic.

    DESIGN PRINCIPLES:

    1. **Atomic Writes**: Save operations should be atomic to prevent corrupted data
    2. **Versioning**: Support multiple versions of benchmark runs
    3. **Queryable**: Enable efficient querying by scale, timestamp, etc.
    4. **Portable**: Results should be serializable (JSON-compatible)

    Implementations:
    - GCSResultStorage: Cloud storage (production)
    - LocalResultStorage: Local filesystem (development)
    - InMemoryResultStorage: Memory only (testing)
    """

    @abstractmethod
    def save_run(self, run_data: Dict[str, Any]) -> str:
        """
        Save a complete benchmark run.

        Args:
            run_data: Dictionary containing all benchmark data:
                - 'run_id': str (unique identifier)
                - 'scale': str (e.g., 'small', 'medium', 'large')
                - 'timestamp': str (ISO format)
                - 'queries': List[Dict] (query data)
                - 'results': List[Dict] (execution results)
                - 'evaluations': List[Dict] (evaluation results)
                - 'metrics': Dict (aggregated metrics)

        Returns:
            The run_id that was saved

        Raises:
            StorageError: If save operation fails

        Note:
            Implementations SHOULD use the provided run_id if present,
            or generate a unique one if not provided.
        """
        pass

    @abstractmethod
    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a benchmark run by ID.

        Args:
            run_id: Unique identifier for the run

        Returns:
            Dictionary with run data if found, None otherwise

        Note:
            The returned dictionary structure MUST match what save_run() accepts.
        """
        pass

    @abstractmethod
    def list_runs(
        self,
        scale: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List recent benchmark runs with optional filtering.

        Args:
            scale: Optional scale filter (e.g., 'small', 'medium')
            limit: Maximum number of runs to return (default 10)

        Returns:
            List of run metadata dictionaries, sorted by timestamp (newest first).
            Each dict contains at minimum:
            - 'run_id': str
            - 'scale': str
            - 'timestamp': str
            - 'summary': Dict (key metrics)

        Note:
            This should return metadata only, not full run data.
            Use load_run() to get complete data.
        """
        pass

    @abstractmethod
    def save_artifact(
        self,
        run_id: str,
        artifact_name: str,
        content: bytes
    ) -> str:
        """
        Save an artifact (report, plot, etc.) associated with a run.

        Args:
            run_id: The run this artifact belongs to
            artifact_name: Name of the artifact (e.g., 'report.md', 'plot.png')
            content: Binary content of the artifact

        Returns:
            URL or path to the saved artifact

        Note:
            Artifacts are supplementary files (reports, visualizations)
            that are linked to a benchmark run.
        """
        pass
