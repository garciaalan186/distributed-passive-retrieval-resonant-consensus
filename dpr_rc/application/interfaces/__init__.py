"""Interfaces - Dependency contracts for use cases"""
from .slm_service import ISLMService
from .router_service import IRouterService
from .worker_service import IWorkerService

__all__ = ["ISLMService", "IRouterService", "IWorkerService"]
