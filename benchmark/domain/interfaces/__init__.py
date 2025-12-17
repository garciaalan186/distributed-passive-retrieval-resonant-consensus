"""
Benchmark Domain Interfaces

These interfaces define the contracts for benchmark components.
All implementations must adhere to these interfaces to ensure:
- Consistent behavior across different implementations
- Easy mocking for testing
- Decoupling from infrastructure details
"""

from .query_executor import (
    IQueryExecutor,
    QueryExecutionResult
)
from .evaluator import IEvaluator
from .result_storage import IResultStorage
from .dataset_generator import (
    IDatasetGenerator,
    BenchmarkDataset
)

__all__ = [
    # Query Execution
    'IQueryExecutor',
    'QueryExecutionResult',

    # Evaluation
    'IEvaluator',

    # Storage
    'IResultStorage',

    # Dataset Generation
    'IDatasetGenerator',
    'BenchmarkDataset',
]
