"""
Benchmark Executors

DPR-RC query executor for local benchmark execution.
"""

from .dprrc_query_executor import DPRRCQueryExecutor, create_dprrc_executor

__all__ = [
    'DPRRCQueryExecutor',
    'create_dprrc_executor',
]
