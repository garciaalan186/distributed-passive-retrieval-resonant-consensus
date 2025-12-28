"""
Benchmark Domain Layer

Core domain interfaces, entities, and value objects for benchmarks.
"""

from .interfaces import (
    IQueryExecutor,
    QueryExecutionResult,
    IEvaluator,
    IResultStorage,
    IDatasetGenerator,
    BenchmarkDataset,
)

__all__ = [
    # Interfaces
    'IQueryExecutor',
    'IEvaluator',
    'IResultStorage',
    'IDatasetGenerator',

    # Data Classes
    'QueryExecutionResult',
    'BenchmarkDataset',
]
