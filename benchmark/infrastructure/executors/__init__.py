"""
Benchmark Executors

Concrete implementations of IQueryExecutor interface.

Available Executors:
- DPRRCQueryExecutor: Executes queries against DPR-RC (currently via HTTP, will use use cases in Phase 2)
- HTTPQueryExecutor: Legacy HTTP executor for cloud deployments
- BaselineExecutor: Executes queries against baseline RAG (PassiveWorker)
"""

from .dprrc_query_executor import DPRRCQueryExecutor, create_dprrc_executor
from .http_query_executor import HTTPQueryExecutor
from .baseline_executor import BaselineExecutor

__all__ = [
    'DPRRCQueryExecutor',
    'create_dprrc_executor',
    'HTTPQueryExecutor',
    'BaselineExecutor',
]
