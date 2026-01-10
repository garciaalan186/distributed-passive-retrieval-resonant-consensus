"""
Benchmark Domain Interfaces

Query executor interface and result dataclass.
"""

from .query_executor import IQueryExecutor, QueryExecutionResult

__all__ = [
    'IQueryExecutor',
    'QueryExecutionResult',
]
