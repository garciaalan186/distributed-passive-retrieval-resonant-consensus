"""
DPR-RC Configuration Module

Provides centralized configuration loading for the DPR-RC system.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import yaml


_config_cache: Optional[Dict[str, Any]] = None
CONFIG_DIR = Path(__file__).parent


def get_dpr_config() -> Dict[str, Any]:
    """
    Load DPR-RC configuration (cached).

    Returns:
        Dict containing all DPR-RC configuration settings.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = CONFIG_DIR / "dpr_rc_config.yaml"
    with open(config_path, 'r') as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


def clear_config_cache() -> None:
    """Clear the config cache (useful for testing)."""
    global _config_cache
    _config_cache = None
