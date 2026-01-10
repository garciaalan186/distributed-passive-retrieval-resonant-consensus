"""
Benchmark Configuration Module

Provides centralized configuration management for benchmark parameters.
Separates scale-independent settings from scale-specific settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Cache for loaded configs
_config_cache: Optional[Dict[str, Any]] = None
_scale_configs_cache: Dict[str, Dict[str, Any]] = {}

# Config directory path
CONFIG_DIR = Path(__file__).parent


def get_config() -> Dict[str, Any]:
    """
    Load scale-independent benchmark configuration.

    Returns cached config on subsequent calls.

    Returns:
        Dict containing all scale-independent settings
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    config_path = CONFIG_DIR / "benchmark_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


def get_scale_config(scale: str) -> Dict[str, Any]:
    """
    Load scale-specific configuration.

    Args:
        scale: Scale name (mini, small, medium, large, stress)

    Returns:
        Dict containing scale-specific settings

    Raises:
        FileNotFoundError: If scale config file doesn't exist
    """
    if scale in _scale_configs_cache:
        return _scale_configs_cache[scale]

    scale_path = CONFIG_DIR / "scales" / f"{scale}.yaml"

    if not scale_path.exists():
        raise FileNotFoundError(f"Scale config not found: {scale_path}")

    with open(scale_path, 'r') as f:
        config = yaml.safe_load(f)

    _scale_configs_cache[scale] = config
    return config


def get_all_scales() -> Dict[str, Dict[str, Any]]:
    """
    Load all scale configurations.

    Returns:
        Dict mapping scale names to their configurations
    """
    scales_dir = CONFIG_DIR / "scales"
    scales = {}

    for scale_file in scales_dir.glob("*.yaml"):
        scale_name = scale_file.stem
        scales[scale_name] = get_scale_config(scale_name)

    return scales


def reload_config() -> None:
    """
    Clear config cache and reload from disk.

    Useful for testing or dynamic config updates.
    """
    global _config_cache, _scale_configs_cache
    _config_cache = None
    _scale_configs_cache = {}


__all__ = [
    'get_config',
    'get_scale_config',
    'get_all_scales',
    'reload_config',
]
