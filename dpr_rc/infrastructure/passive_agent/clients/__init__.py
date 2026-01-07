"""
Infrastructure Clients for Passive Agent
"""

from .http_slm_client import HttpSLMClient
from .direct_slm_client import DirectSLMClient

__all__ = ["HttpSLMClient", "DirectSLMClient"]
