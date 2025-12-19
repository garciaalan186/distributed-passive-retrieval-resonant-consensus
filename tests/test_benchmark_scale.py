"""
Test to verify BENCHMARK_SCALE environment variable functionality.
"""

import pytest
import os
import sys
from pathlib import Path

# Add benchmark directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.research_benchmark import ResearchBenchmarkSuite


def test_small_only_scale():
    """Test that BENCHMARK_SCALE=small only runs small scale"""
    # Set environment variable
    os.environ["BENCHMARK_SCALE"] = "small"

    # Initialize benchmark suite
    suite = ResearchBenchmarkSuite(output_dir="test_scale_results")

    # Verify only small scale is configured
    assert len(suite.scale_levels) == 1
    assert suite.scale_levels[0]["name"] == "small"
    assert suite.scale_levels[0]["events_per_topic_per_year"] == 10
    assert suite.scale_levels[0]["num_domains"] == 2

    # Clean up
    del os.environ["BENCHMARK_SCALE"]


def test_multiple_scales():
    """Test that BENCHMARK_SCALE=small,medium runs both scales"""
    os.environ["BENCHMARK_SCALE"] = "small,medium"

    suite = ResearchBenchmarkSuite(output_dir="test_scale_results")

    # Verify both scales are configured
    assert len(suite.scale_levels) == 2
    scale_names = [s["name"] for s in suite.scale_levels]
    assert "small" in scale_names
    assert "medium" in scale_names

    del os.environ["BENCHMARK_SCALE"]


def test_all_scales():
    """Test that BENCHMARK_SCALE=all runs all 4 scales"""
    os.environ["BENCHMARK_SCALE"] = "all"

    suite = ResearchBenchmarkSuite(output_dir="test_scale_results")

    # Verify all scales are configured
    assert len(suite.scale_levels) == 4
    scale_names = [s["name"] for s in suite.scale_levels]
    assert "small" in scale_names
    assert "medium" in scale_names
    assert "large" in scale_names
    assert "stress" in scale_names

    del os.environ["BENCHMARK_SCALE"]


def test_invalid_scale_falls_back_to_small():
    """Test that invalid scale falls back to small"""
    os.environ["BENCHMARK_SCALE"] = "invalid_scale_name"

    suite = ResearchBenchmarkSuite(output_dir="test_scale_results")

    # Should fall back to small
    assert len(suite.scale_levels) == 1
    assert suite.scale_levels[0]["name"] == "small"

    del os.environ["BENCHMARK_SCALE"]


def test_default_scale_is_all():
    """Test that default (no env var) is all scales"""
    # Ensure env var is not set
    if "BENCHMARK_SCALE" in os.environ:
        del os.environ["BENCHMARK_SCALE"]

    suite = ResearchBenchmarkSuite(output_dir="test_scale_results")

    # Default should be all scales
    assert len(suite.scale_levels) == 4


def test_scale_config_parameters():
    """Test that each scale has correct parameters"""
    os.environ["BENCHMARK_SCALE"] = "all"

    suite = ResearchBenchmarkSuite(output_dir="test_scale_results")

    # Find each scale and verify its config
    configs = {s["name"]: s for s in suite.scale_levels}

    assert configs["small"]["events_per_topic_per_year"] == 10
    assert configs["small"]["num_domains"] == 2

    assert configs["medium"]["events_per_topic_per_year"] == 25
    assert configs["medium"]["num_domains"] == 3

    assert configs["large"]["events_per_topic_per_year"] == 50
    assert configs["large"]["num_domains"] == 4

    assert configs["stress"]["events_per_topic_per_year"] == 100
    assert configs["stress"]["num_domains"] == 5

    del os.environ["BENCHMARK_SCALE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
