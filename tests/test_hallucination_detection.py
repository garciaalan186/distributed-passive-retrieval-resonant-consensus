"""
Unit tests for SLM-based hallucination detection.

Tests cover:
1. SLM service integration
2. Fallback detection logic
3. Retry mechanisms
4. Error handling
5. Response parsing
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add benchmark directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.research_benchmark import ResearchBenchmarkSuite


@pytest.fixture
def benchmark_suite():
    """Create a benchmark suite instance for testing"""
    return ResearchBenchmarkSuite(output_dir="test_benchmark_results")


@pytest.fixture
def sample_glossary():
    """Sample glossary matching actual structure"""
    return {
        'physics': {
            'particles': {
                'Blarkon': {'type': 'fermion', 'charge': 1},
                'Zorptex': {'type': 'boson', 'charge': 0},
                'Quixel': {'type': 'lepton', 'charge': -1}
            },
            'phenomena': {
                'Fluctron': {'type': 'quantum_effect'},
                'Resonex': {'type': 'field_effect'}
            },
            'constants': {
                'Veltrix': {'type': 'speed_limit', 'value': 3e8}
            }
        },
        'domains': {
            'Chronophysics': {
                'concepts': {
                    'TimeDilation': {'description': 'time warping'},
                    'CausalLoop': {'description': 'time loop'}
                }
            }
        }
    }


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth data"""
    return {
        'expected_consensus': ['Blarkon decay rates improved'],
        'expected_disputed': ['Zorptex stability uncertain']
    }


# =============================================================================
# SLM SERVICE INTEGRATION TESTS
# =============================================================================

def test_slm_detection_with_valid_response(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test SLM detection with properly formatted response"""
    with patch('requests.post') as mock_post:
        # Mock successful SLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "has_hallucination": False,
            "hallucination_type": None,
            "explanation": "All terms valid",
            "severity": "none",
            "flagged_content": []
        }
        mock_post.return_value = mock_response

        result = benchmark_suite.detect_hallucination_via_slm(
            query="What is Blarkon status?",
            ground_truth=sample_ground_truth,
            system_response="Blarkon research shows progress in decay rates.",
            glossary=sample_glossary,
            confidence=0.8
        )

        assert result["has_hallucination"] == False
        assert result["severity"] == "none"
        assert mock_post.call_count == 1  # No retries needed


def test_slm_detection_identifies_hallucination(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that SLM correctly identifies hallucinations"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "has_hallucination": True,
            "hallucination_type": "invalid_term",
            "explanation": "Zynthium not in glossary",
            "severity": "high",
            "flagged_content": ["Zynthium"]
        }
        mock_post.return_value = mock_response

        result = benchmark_suite.detect_hallucination_via_slm(
            query="What is Blarkon status?",
            ground_truth=sample_ground_truth,
            system_response="Zynthium interference was solved.",  # Zynthium not in glossary
            glossary=sample_glossary,
            confidence=0.9
        )

        assert result["has_hallucination"] == True
        assert result["hallucination_type"] == "invalid_term"
        assert result["severity"] == "high"
        assert "Zynthium" in result["flagged_content"]


# =============================================================================
# RETRY LOGIC TESTS
# =============================================================================

def test_retry_on_server_error(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that 5xx errors trigger retries"""
    with patch('requests.post') as mock_post, \
         patch('time.sleep') as mock_sleep:  # Don't actually sleep

        # First two attempts fail with 503, third succeeds
        mock_responses = [
            Mock(status_code=503),
            Mock(status_code=503),
            Mock(status_code=200, json=lambda: {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "Success after retry",
                "severity": "none",
                "flagged_content": []
            })
        ]
        mock_post.side_effect = mock_responses

        result = benchmark_suite.detect_hallucination_via_slm(
            query="Test query",
            ground_truth=sample_ground_truth,
            system_response="Test response",
            glossary=sample_glossary,
            confidence=0.7
        )

        assert result["has_hallucination"] == False
        assert mock_post.call_count == 3  # 2 failures + 1 success
        assert mock_sleep.call_count == 2  # Slept between retries


def test_retry_on_timeout(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that timeouts trigger retries"""
    import requests

    with patch('requests.post') as mock_post, \
         patch('time.sleep') as mock_sleep:

        # First two attempts timeout, third succeeds
        mock_post.side_effect = [
            requests.Timeout("timeout 1"),
            requests.Timeout("timeout 2"),
            Mock(status_code=200, json=lambda: {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "Success after timeout retries",
                "severity": "none",
                "flagged_content": []
            })
        ]

        result = benchmark_suite.detect_hallucination_via_slm(
            query="Test query",
            ground_truth=sample_ground_truth,
            system_response="Test response",
            glossary=sample_glossary,
            confidence=0.7
        )

        assert result["has_hallucination"] == False
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2


def test_no_retry_on_client_error(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that 4xx errors don't trigger retries"""
    with patch('requests.post') as mock_post, \
         patch('time.sleep') as mock_sleep:

        # 400 error should not retry
        mock_response = Mock(status_code=400)
        mock_post.return_value = mock_response

        result = benchmark_suite.detect_hallucination_via_slm(
            query="Test query",
            ground_truth=sample_ground_truth,
            system_response="Test response",
            glossary=sample_glossary,
            confidence=0.7
        )

        # Should fall back to rule-based detection
        assert mock_post.call_count == 1  # No retries
        assert mock_sleep.call_count == 0  # No sleep


def test_exponential_backoff_timing(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that retry delays follow exponential backoff"""
    import requests

    with patch('requests.post') as mock_post, \
         patch('time.sleep') as mock_sleep:

        # All attempts fail
        mock_post.side_effect = requests.Timeout("always timeout")

        result = benchmark_suite.detect_hallucination_via_slm(
            query="Test query",
            ground_truth=sample_ground_truth,
            system_response="Test response",
            glossary=sample_glossary,
            confidence=0.7
        )

        # Should fall back after 3 attempts
        assert mock_post.call_count == 3

        # Check exponential backoff: 1s, 2s (base_delay * 2^attempt)
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 1.0  # First retry: 1s
        assert calls[1][0][0] == 2.0  # Second retry: 2s


# =============================================================================
# FALLBACK DETECTION TESTS
# =============================================================================

def test_fallback_detection_common_words_not_flagged(benchmark_suite, sample_glossary):
    """Test that common English words are not flagged as hallucinations"""
    response = "The research shows No significant progress in the analysis."

    result = benchmark_suite._fallback_hallucination_detection(
        response=response,
        glossary=sample_glossary,
        confidence=0.9
    )

    assert result["has_hallucination"] == False
    assert result["severity"] == "none"
    # Common words like "The", "No", "Research" should not be flagged


def test_fallback_detection_invalid_terms_flagged(benchmark_suite, sample_glossary):
    """Test that truly invalid terms are flagged"""
    response = "Zynthium particles caused XyloBreak cascade effects."
    # "Zynthium" and "XyloBreak" are not in glossary

    result = benchmark_suite._fallback_hallucination_detection(
        response=response,
        glossary=sample_glossary,
        confidence=0.9
    )

    assert result["has_hallucination"] == True
    assert result["hallucination_type"] == "invalid_term"
    assert "Zynthium" in result["flagged_content"]
    assert "XyloBreak" in result["flagged_content"]


def test_fallback_detection_valid_glossary_terms_not_flagged(benchmark_suite, sample_glossary):
    """Test that valid glossary terms are not flagged"""
    response = "Blarkon and Zorptex interactions show Fluctron effects."
    # All these terms are in the glossary

    result = benchmark_suite._fallback_hallucination_detection(
        response=response,
        glossary=sample_glossary,
        confidence=0.8
    )

    assert result["has_hallucination"] == False


def test_fallback_lenient_with_uncertainty(benchmark_suite, sample_glossary):
    """Test lenient handling of uncertain responses"""
    response = "Results are mixed. Blarkon or FakeParticle may be involved."
    # "FakeParticle" not in glossary, but response shows uncertainty

    result = benchmark_suite._fallback_hallucination_detection(
        response=response,
        glossary=sample_glossary,
        confidence=0.5  # Low confidence
    )

    # Should be lenient due to low confidence + uncertainty language
    # A single suspicious term with low confidence shouldn't trigger
    assert result["has_hallucination"] == False


def test_fallback_strict_with_high_confidence(benchmark_suite, sample_glossary):
    """Test strict handling when system is confident"""
    response = "FakeParticle definitively causes quantum effects."

    result = benchmark_suite._fallback_hallucination_detection(
        response=response,
        glossary=sample_glossary,
        confidence=0.95  # High confidence
    )

    # High confidence + invalid term should be flagged
    assert result["has_hallucination"] == True
    assert "FakeParticle" in result["flagged_content"]


# =============================================================================
# GLOSSARY PARSING TESTS
# =============================================================================

def test_glossary_structure_parsing(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that glossary terms are correctly extracted"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "has_hallucination": False,
            "hallucination_type": None,
            "explanation": "Test",
            "severity": "none",
            "flagged_content": []
        }
        mock_post.return_value = mock_response

        benchmark_suite.detect_hallucination_via_slm(
            query="Test",
            ground_truth=sample_ground_truth,
            system_response="Test",
            glossary=sample_glossary,
            confidence=0.7
        )

        # Check that request included valid_terms
        call_args = mock_post.call_args
        request_json = call_args[1]['json']
        valid_terms = request_json['valid_terms']

        # Should include particles
        assert 'Blarkon' in valid_terms
        assert 'Zorptex' in valid_terms

        # Should include phenomena
        assert 'Fluctron' in valid_terms or 'Resonex' in valid_terms

        # Should include domain concepts
        assert 'TimeDilation' in valid_terms or 'CausalLoop' in valid_terms


def test_empty_glossary_handled_gracefully(benchmark_suite, sample_ground_truth):
    """Test that empty glossary doesn't crash"""
    empty_glossary = {}

    result = benchmark_suite._fallback_hallucination_detection(
        response="Some text",
        glossary=empty_glossary,
        confidence=0.7
    )

    # Should not crash, should handle gracefully
    assert "has_hallucination" in result


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_malformed_json_response_handled(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test handling of malformed JSON from SLM"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        # Missing required fields
        mock_response.json.return_value = {
            "has_hallucination": True
            # Missing other fields
        }
        mock_post.return_value = mock_response

        result = benchmark_suite.detect_hallucination_via_slm(
            query="Test",
            ground_truth=sample_ground_truth,
            system_response="Test",
            glossary=sample_glossary,
            confidence=0.7
        )

        # Should handle gracefully with defaults
        assert "has_hallucination" in result
        assert "severity" in result


def test_connection_error_triggers_fallback(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that connection errors trigger fallback"""
    import requests

    with patch('requests.post') as mock_post, \
         patch('time.sleep'):

        # All attempts fail with connection error
        mock_post.side_effect = requests.ConnectionError("Network unreachable")

        result = benchmark_suite.detect_hallucination_via_slm(
            query="Test",
            ground_truth=sample_ground_truth,
            system_response="Test contains FakeTerm",
            glossary=sample_glossary,
            confidence=0.7
        )

        # Should fall back to rule-based detection
        assert "has_hallucination" in result
        # Fallback should detect "FakeTerm"
        assert result["has_hallucination"] == True or "FakeTerm" in str(result)


def test_slm_service_url_none_uses_fallback(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test that None SLM service URL goes straight to fallback"""
    benchmark_suite.slm_service_url = None

    result = benchmark_suite.detect_hallucination_via_slm(
        query="Test",
        ground_truth=sample_ground_truth,
        system_response="Valid Blarkon response",
        glossary=sample_glossary,
        confidence=0.7
    )

    # Should use fallback without attempting SLM call
    assert "has_hallucination" in result


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_compare_results_uses_hallucination_detection(benchmark_suite, sample_glossary):
    """Test that compare_results integrates hallucination detection"""
    queries = [
        {
            "question": "What is Blarkon status?",
            "expected_consensus": ["Blarkon stable"],
            "expected_disputed": []
        }
    ]

    dprrc_results = [
        {
            "success": True,
            "response": "Blarkon shows stability.",
            "confidence": 0.8,
            "latency_ms": 100,
            "query_id": "query_0001"
        }
    ]

    baseline_results = [
        {
            "success": True,
            "response": "Blarkon shows stability.",
            "confidence": 1.0,
            "latency_ms": 50,
            "query_id": "query_0001"
        }
    ]

    # Now using batch detection - mock the batch method instead
    with patch.object(benchmark_suite, 'batch_detect_hallucination_via_slm') as mock_batch:
        mock_batch.return_value = [
            {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "All valid",
                "severity": "none",
                "flagged_content": []
            },
            {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "All valid",
                "severity": "none",
                "flagged_content": []
            }
        ]

        result = benchmark_suite.compare_results(
            queries=queries,
            dprrc_results=dprrc_results,
            baseline_results=baseline_results,
            glossary=sample_glossary
        )

        # Should have called batch detection once (not 2 individual calls)
        assert mock_batch.call_count == 1
        # Batch should have received 2 requests (1 dprrc + 1 baseline)
        call_args = mock_batch.call_args[0][0]
        assert len(call_args) == 2
        assert result["dprrc_hallucination_count"] == 0
        assert result["baseline_hallucination_count"] == 0


# =============================================================================
# BATCH HALLUCINATION DETECTION TESTS
# =============================================================================

def test_batch_hallucination_detection(benchmark_suite, sample_glossary, sample_ground_truth):
    """Test batch hallucination detection with multiple requests"""
    with patch('requests.post') as mock_post:
        # Mock batch response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "trace_id": "query_001",
                    "has_hallucination": False,
                    "hallucination_type": None,
                    "explanation": "All valid",
                    "severity": "none",
                    "flagged_content": []
                },
                {
                    "trace_id": "query_002",
                    "has_hallucination": True,
                    "hallucination_type": "invalid_term",
                    "explanation": "FakeTerm not in glossary",
                    "severity": "high",
                    "flagged_content": ["FakeTerm"]
                }
            ],
            "model_id": "Qwen/Qwen2-0.5B-Instruct",
            "batch_size": 2
        }
        mock_post.return_value = mock_response

        # Create batch requests
        check_requests = [
            {
                "query": "What is Blarkon?",
                "system_response": "Blarkon is stable",
                "ground_truth": sample_ground_truth,
                "valid_terms": ["Blarkon", "Zorptex"],
                "confidence": 0.8,
                "trace_id": "query_001"
            },
            {
                "query": "What is FakeTerm?",
                "system_response": "FakeTerm causes quantum effects",
                "ground_truth": sample_ground_truth,
                "valid_terms": ["Blarkon", "Zorptex"],
                "confidence": 0.9,
                "trace_id": "query_002"
            }
        ]

        results = benchmark_suite.batch_detect_hallucination_via_slm(check_requests)

        # Verify single batch call was made
        assert mock_post.call_count == 1

        # Verify results
        assert len(results) == 2
        assert results[0]["has_hallucination"] == False
        assert results[1]["has_hallucination"] == True
        assert "FakeTerm" in results[1]["flagged_content"]


def test_batch_hallucination_empty_requests(benchmark_suite):
    """Test batch detection with empty request list"""
    results = benchmark_suite.batch_detect_hallucination_via_slm([])
    assert results == []


def test_batch_hallucination_fallback_on_error(benchmark_suite, sample_glossary):
    """Test that batch detection falls back to rule-based on error"""
    with patch('requests.post') as mock_post:
        # Mock HTTP error
        mock_post.side_effect = Exception("Network error")

        check_requests = [
            {
                "query": "Test",
                "system_response": "FakeTerm is real",
                "ground_truth": {},
                "valid_terms": ["Blarkon"],
                "confidence": 0.9,
                "trace_id": "query_001"
            }
        ]

        results = benchmark_suite.batch_detect_hallucination_via_slm(check_requests)

        # Should get fallback results
        assert len(results) == 1
        assert "has_hallucination" in results[0]


def test_batch_hallucination_no_slm_url(benchmark_suite):
    """Test batch detection when SLM URL is not configured"""
    benchmark_suite.slm_service_url = None

    check_requests = [
        {
            "query": "Test",
            "system_response": "Valid response",
            "ground_truth": {},
            "valid_terms": ["Valid"],
            "confidence": 0.8,
            "trace_id": "query_001"
        }
    ]

    results = benchmark_suite.batch_detect_hallucination_via_slm(check_requests)

    # Should use fallback without attempting HTTP call
    assert len(results) == 1
    assert "has_hallucination" in results[0]


def test_compare_results_uses_batch_detection(benchmark_suite, sample_glossary):
    """Test that compare_results uses batch hallucination detection"""
    queries = [
        {
            "question": "What is Blarkon?",
            "expected_consensus": ["Blarkon stable"],
            "expected_disputed": []
        },
        {
            "question": "What is Zorptex?",
            "expected_consensus": ["Zorptex unstable"],
            "expected_disputed": []
        }
    ]

    dprrc_results = [
        {
            "success": True,
            "response": "Blarkon is stable",
            "confidence": 0.8,
            "latency_ms": 100,
            "query_id": "dprrc_001"
        },
        {
            "success": True,
            "response": "Zorptex is unstable",
            "confidence": 0.9,
            "latency_ms": 120,
            "query_id": "dprrc_002"
        }
    ]

    baseline_results = [
        {
            "success": True,
            "response": "Blarkon is stable",
            "confidence": 1.0,
            "latency_ms": 50,
            "query_id": "baseline_001"
        },
        {
            "success": True,
            "response": "Zorptex is unstable",
            "confidence": 1.0,
            "latency_ms": 60,
            "query_id": "baseline_002"
        }
    ]

    with patch.object(benchmark_suite, 'batch_detect_hallucination_via_slm') as mock_batch:
        mock_batch.return_value = [
            {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "Valid",
                "severity": "none",
                "flagged_content": []
            },
            {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "Valid",
                "severity": "none",
                "flagged_content": []
            },
            {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "Valid",
                "severity": "none",
                "flagged_content": []
            },
            {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "Valid",
                "severity": "none",
                "flagged_content": []
            }
        ]

        result = benchmark_suite.compare_results(
            queries=queries,
            dprrc_results=dprrc_results,
            baseline_results=baseline_results,
            glossary=sample_glossary
        )

        # Should have called batch detection once (not 4 individual calls)
        assert mock_batch.call_count == 1

        # Verify the batch request included all 4 checks (2 dprrc + 2 baseline)
        call_args = mock_batch.call_args[0][0]
        assert len(call_args) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
