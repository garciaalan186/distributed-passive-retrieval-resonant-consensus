"""
Comprehensive regression tests for batch SLM verification implementation.

Tests:
1. API Contract validation
2. Backward compatibility
3. Fallback behavior
4. Result ordering preservation
5. Edge cases
6. Integration with compare_results
7. Performance characteristics
8. Error handling
"""

import pytest
from unittest.mock import Mock, patch
from benchmark.research_benchmark import ResearchBenchmarkSuite


@pytest.fixture
def benchmark_suite():
    """Create benchmark suite for testing"""
    suite = ResearchBenchmarkSuite(output_dir="test_output")
    suite.slm_service_url = "http://localhost:8081"
    return suite


@pytest.fixture
def sample_glossary():
    """Sample glossary for testing"""
    return {
        "physics": {
            "particles": {
                "Blarkon": {"stability": "stable"},
                "Zorptex": {"stability": "unstable"}
            },
            "phenomena": {
                "Quantum Entanglement": {}
            }
        },
        "domains": {
            "particle_physics": {
                "concepts": {
                    "Superposition": {},
                    "Wave Function": {}
                }
            }
        }
    }


class TestAPIContract:
    """Test API contract compliance"""

    def test_batch_endpoint_request_structure(self, benchmark_suite):
        """Batch endpoint accepts list of HallucinationCheckRequest"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "trace_id": "test_001",
                        "has_hallucination": False,
                        "hallucination_type": None,
                        "explanation": "Valid",
                        "severity": "none",
                        "flagged_content": []
                    }
                ],
                "model_id": "test-model",
                "batch_size": 1
            }
            mock_post.return_value = mock_response

            requests = [{
                "query": "Test query",
                "system_response": "Test response",
                "ground_truth": {},
                "valid_terms": ["Test"],
                "confidence": 0.8,
                "trace_id": "test_001"
            }]

            benchmark_suite.batch_detect_hallucination_via_slm(requests)

            # Verify endpoint called correctly
            assert mock_post.called
            call_args = mock_post.call_args
            assert "/batch_check_hallucination" in call_args[0][0]
            assert call_args[1]["json"] == requests

    def test_batch_response_structure(self, benchmark_suite):
        """Batch endpoint returns expected response structure"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "trace_id": "test_001",
                        "has_hallucination": True,
                        "hallucination_type": "invalid_term",
                        "explanation": "Test explanation",
                        "severity": "high",
                        "flagged_content": ["FakeTerm"]
                    }
                ],
                "model_id": "Qwen/Qwen2-0.5B-Instruct",
                "batch_size": 1
            }
            mock_post.return_value = mock_response

            requests = [{
                "query": "Test",
                "system_response": "FakeTerm is real",
                "ground_truth": {},
                "valid_terms": [],
                "confidence": 0.9,
                "trace_id": "test_001"
            }]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            # Verify response structure
            assert len(results) == 1
            result = results[0]
            assert "has_hallucination" in result
            assert "hallucination_type" in result
            assert "explanation" in result
            assert "severity" in result
            assert "flagged_content" in result
            assert result["has_hallucination"] == True
            assert result["hallucination_type"] == "invalid_term"


class TestBackwardCompatibility:
    """Test that individual detection still works"""

    def test_individual_detection_still_works(self, benchmark_suite, sample_glossary):
        """Original individual detection method still functions"""
        # The individual method is _fallback_hallucination_detection
        result = benchmark_suite._fallback_hallucination_detection(
            response="FakeTerm is real",
            glossary=sample_glossary,
            confidence=0.9
        )

        assert "has_hallucination" in result
        assert "hallucination_type" in result

    def test_single_item_batch_behaves_like_individual(self, benchmark_suite):
        """Single-item batch should behave identically to individual call"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [{
                    "trace_id": "test_001",
                    "has_hallucination": False,
                    "hallucination_type": None,
                    "explanation": "Valid",
                    "severity": "none",
                    "flagged_content": []
                }],
                "model_id": "test-model",
                "batch_size": 1
            }
            mock_post.return_value = mock_response

            requests = [{
                "query": "Test",
                "system_response": "Valid response",
                "ground_truth": {},
                "valid_terms": ["Valid"],
                "confidence": 0.8,
                "trace_id": "test_001"
            }]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            assert len(results) == 1
            assert results[0]["has_hallucination"] == False


class TestFallbackBehavior:
    """Test graceful degradation"""

    def test_http_error_triggers_fallback(self, benchmark_suite):
        """HTTP errors should trigger fallback detection"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response

            requests = [{
                "query": "Test",
                "system_response": "Test response",
                "ground_truth": {},
                "valid_terms": ["Test"],
                "confidence": 0.8,
                "trace_id": "test_001"
            }]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            # Should get fallback results
            assert len(results) == 1
            assert "has_hallucination" in results[0]

    def test_network_error_triggers_fallback(self, benchmark_suite):
        """Network errors should trigger fallback detection"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Connection refused")

            requests = [{
                "query": "Test",
                "system_response": "Test response",
                "ground_truth": {},
                "valid_terms": ["Test"],
                "confidence": 0.8,
                "trace_id": "test_001"
            }]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            # Should get fallback results
            assert len(results) == 1
            assert "has_hallucination" in results[0]

    def test_no_slm_url_uses_fallback_directly(self, benchmark_suite):
        """When SLM URL is None, should use fallback without HTTP call"""
        benchmark_suite.slm_service_url = None

        with patch('requests.post') as mock_post:
            requests = [{
                "query": "Test",
                "system_response": "Test response",
                "ground_truth": {},
                "valid_terms": ["Test"],
                "confidence": 0.8,
                "trace_id": "test_001"
            }]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            # Should NOT have made HTTP call
            assert not mock_post.called
            # Should have results
            assert len(results) == 1


class TestResultOrdering:
    """Test that batch results maintain correct order"""

    def test_result_order_preserved(self, benchmark_suite):
        """Results should be returned in same order as requests"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "trace_id": "first",
                        "has_hallucination": True,
                        "hallucination_type": "invalid_term",
                        "explanation": "First result",
                        "severity": "high",
                        "flagged_content": ["Bad1"]
                    },
                    {
                        "trace_id": "second",
                        "has_hallucination": False,
                        "hallucination_type": None,
                        "explanation": "Second result",
                        "severity": "none",
                        "flagged_content": []
                    },
                    {
                        "trace_id": "third",
                        "has_hallucination": True,
                        "hallucination_type": "fabricated_fact",
                        "explanation": "Third result",
                        "severity": "medium",
                        "flagged_content": ["Bad2"]
                    }
                ],
                "model_id": "test-model",
                "batch_size": 3
            }
            mock_post.return_value = mock_response

            requests = [
                {
                    "query": "Q1",
                    "system_response": "R1",
                    "ground_truth": {},
                    "valid_terms": [],
                    "confidence": 0.8,
                    "trace_id": "first"
                },
                {
                    "query": "Q2",
                    "system_response": "R2",
                    "ground_truth": {},
                    "valid_terms": [],
                    "confidence": 0.9,
                    "trace_id": "second"
                },
                {
                    "query": "Q3",
                    "system_response": "R3",
                    "ground_truth": {},
                    "valid_terms": [],
                    "confidence": 0.7,
                    "trace_id": "third"
                }
            ]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            # Verify order
            assert len(results) == 3
            assert results[0]["has_hallucination"] == True
            assert results[0]["explanation"] == "First result"
            assert results[1]["has_hallucination"] == False
            assert results[1]["explanation"] == "Second result"
            assert results[2]["has_hallucination"] == True
            assert results[2]["explanation"] == "Third result"

    def test_compare_results_maps_correctly(self, benchmark_suite, sample_glossary):
        """compare_results should correctly map batch results to dprrc/baseline"""
        queries = [
            {
                "question": "What is Blarkon?",
                "expected_consensus": ["stable"],
                "expected_disputed": []
            },
            {
                "question": "What is Zorptex?",
                "expected_consensus": ["unstable"],
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
                "response": "Zorptex has invalid FakeTerm",  # Hallucination
                "confidence": 0.7,
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
            # Mock returns in INTERLEAVED order: dprrc_001 (ok), baseline_001 (ok), dprrc_002 (hallucination), baseline_002 (ok)
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
                    "has_hallucination": True,
                    "hallucination_type": "invalid_term",
                    "explanation": "FakeTerm not valid",
                    "severity": "high",
                    "flagged_content": ["FakeTerm"]
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

            # Verify hallucination correctly mapped to dprrc
            assert result["dprrc_hallucination_count"] == 1
            assert result["baseline_hallucination_count"] == 0
            assert len(result["dprrc_hallucination_details"]) == 1
            assert result["dprrc_hallucination_details"][0]["query_id"] == "dprrc_002"


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_batch(self, benchmark_suite):
        """Empty batch should return empty results"""
        results = benchmark_suite.batch_detect_hallucination_via_slm([])
        assert results == []

    def test_mixed_success_failure_in_batch(self):
        """Test SLM service handling of mixed success/failure"""
        # This tests the server-side batch endpoint error handling
        # Server should return partial success for errors
        pass  # Covered by server-side tests

    def test_large_batch(self, benchmark_suite):
        """Test with large batch to ensure no overflow issues"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            # Create 100 results
            mock_response.json.return_value = {
                "results": [
                    {
                        "trace_id": f"test_{i}",
                        "has_hallucination": False,
                        "hallucination_type": None,
                        "explanation": "Valid",
                        "severity": "none",
                        "flagged_content": []
                    }
                    for i in range(100)
                ],
                "model_id": "test-model",
                "batch_size": 100
            }
            mock_post.return_value = mock_response

            requests = [
                {
                    "query": f"Q{i}",
                    "system_response": f"R{i}",
                    "ground_truth": {},
                    "valid_terms": [],
                    "confidence": 0.8,
                    "trace_id": f"test_{i}"
                }
                for i in range(100)
            ]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            assert len(results) == 100


class TestPerformance:
    """Test performance characteristics"""

    def test_batching_reduces_http_calls(self, benchmark_suite, sample_glossary):
        """Verify batching makes single HTTP call instead of N calls"""
        queries = [
            {
                "question": f"Question {i}",
                "expected_consensus": ["answer"],
                "expected_disputed": []
            }
            for i in range(10)
        ]

        dprrc_results = [
            {
                "success": True,
                "response": f"Response {i}",
                "confidence": 0.8,
                "latency_ms": 100,
                "query_id": f"dprrc_{i}"
            }
            for i in range(10)
        ]

        baseline_results = [
            {
                "success": True,
                "response": f"Response {i}",
                "confidence": 1.0,
                "latency_ms": 50,
                "query_id": f"baseline_{i}"
            }
            for i in range(10)
        ]

        with patch.object(benchmark_suite, 'batch_detect_hallucination_via_slm') as mock_batch:
            mock_batch.return_value = [
                {
                    "has_hallucination": False,
                    "hallucination_type": None,
                    "explanation": "Valid",
                    "severity": "none",
                    "flagged_content": []
                }
                for _ in range(20)  # 10 dprrc + 10 baseline
            ]

            benchmark_suite.compare_results(
                queries=queries,
                dprrc_results=dprrc_results,
                baseline_results=baseline_results,
                glossary=sample_glossary
            )

            # Should be called ONCE with 20 requests (10 dprrc + 10 baseline)
            assert mock_batch.call_count == 1
            call_args = mock_batch.call_args[0][0]
            assert len(call_args) == 20


class TestErrorHandling:
    """Test error handling"""

    def test_malformed_response_doesnt_crash(self, benchmark_suite):
        """Malformed response should trigger fallback"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "invalid": "structure"
                # Missing "results" key
            }
            mock_post.return_value = mock_response

            requests = [{
                "query": "Test",
                "system_response": "Test",
                "ground_truth": {},
                "valid_terms": [],
                "confidence": 0.8,
                "trace_id": "test_001"
            }]

            results = benchmark_suite.batch_detect_hallucination_via_slm(requests)

            # Should get fallback results instead of crashing
            assert len(results) == 1

    def test_timeout_triggers_fallback(self, benchmark_suite):
        """Timeout should trigger fallback"""
        with patch('requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.Timeout("Request timeout")

            reqs = [{
                "query": "Test",
                "system_response": "Test",
                "ground_truth": {},
                "valid_terms": [],
                "confidence": 0.8,
                "trace_id": "test_001"
            }]

            results = benchmark_suite.batch_detect_hallucination_via_slm(reqs)

            # Should get fallback results
            assert len(results) == 1
            assert "has_hallucination" in results[0]

    def test_partial_results_dont_corrupt_benchmark(self, benchmark_suite, sample_glossary):
        """Partial/incomplete results shouldn't corrupt benchmark stats"""
        queries = [
            {
                "question": "Q1",
                "expected_consensus": ["answer"],
                "expected_disputed": []
            }
        ]

        dprrc_results = [
            {
                "success": True,
                "response": "R1",
                "confidence": 0.8,
                "latency_ms": 100,
                "query_id": "dprrc_001"
            }
        ]

        baseline_results = [
            {
                "success": True,
                "response": "R1",
                "confidence": 1.0,
                "latency_ms": 50,
                "query_id": "baseline_001"
            }
        ]

        with patch.object(benchmark_suite, 'batch_detect_hallucination_via_slm') as mock_batch:
            # Return fewer results than expected (error case)
            mock_batch.return_value = [
                {
                    "has_hallucination": False,
                    "hallucination_type": None,
                    "explanation": "Valid",
                    "severity": "none",
                    "flagged_content": []
                }
                # Only 1 result instead of 2
            ]

            # Should not crash
            result = benchmark_suite.compare_results(
                queries=queries,
                dprrc_results=dprrc_results,
                baseline_results=baseline_results,
                glossary=sample_glossary
            )

            # Stats should still be valid
            assert "total_queries" in result
            assert "dprrc_hallucination_count" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
