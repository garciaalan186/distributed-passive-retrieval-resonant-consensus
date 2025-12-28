"""
Application Layer for Passive Agent

Coordinates domain services and infrastructure.
Contains use cases (application orchestration) and DTOs.
"""

from .dtos import ProcessRFIRequest, ProcessRFIResponse
from .use_cases import ProcessRFIUseCase, ILogger

__all__ = [
    "ProcessRFIRequest",
    "ProcessRFIResponse",
    "ProcessRFIUseCase",
    "ILogger",
]
