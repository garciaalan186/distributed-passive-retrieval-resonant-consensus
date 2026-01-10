"""
Benchmark Domain Layer

Interfaces, services, and value objects for benchmark evaluation.
"""

from .interfaces import IQueryExecutor, QueryExecutionResult
from .services import EvaluationService
from .value_objects import CorrectnessResult

__all__ = [
    'IQueryExecutor',
    'QueryExecutionResult',
    'EvaluationService',
    'CorrectnessResult',
]
