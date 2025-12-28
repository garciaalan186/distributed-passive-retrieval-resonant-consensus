"""
DPR-RC Research-Grade Benchmark Suite
Implements peer-review-ready evaluation with complete audit trails
"""

# Python 3.9 compatibility: patch importlib.metadata for chromadb
import importlib.metadata
if not hasattr(importlib.metadata, 'packages_distributions'):
    def _packages_distributions():
        return {}
    importlib.metadata.packages_distributions = _packages_distributions

import os
import json
import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import requests

from .synthetic_history import SyntheticHistoryGeneratorV2
from benchmark.domain.services import EvaluationService


@dataclass
class SuperpositionEvaluation:
    """Evaluation for DPR-RC superposition responses"""
    correct_answer_present: bool
    correct_in_consensus: bool
    correct_in_perspectival: bool
    ideal_placement: bool
    multiple_alternatives_presented: bool


@dataclass
class HallucinationAnalysis:
    total_claims: int
    hallucinated_claims: int
    hallucination_rate: float
    legitimate_alternatives: int
    details: List[Dict]


@dataclass
class ResourceMetrics:
    mean_latency_ms: float
    p95_latency_ms: float
    tokens_per_query: float
    estimated_cost_usd: float


class ResearchBenchmarkSuite:
    """
    Publication-ready benchmark with complete audit trails
    """
    
    def __init__(self, output_dir: str = "benchmark_results_research"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # All available scale levels
        all_scales = {
            "small": {"name": "small", "events_per_topic_per_year": 10, "num_domains": 2},
            "medium": {"name": "medium", "events_per_topic_per_year": 25, "num_domains": 3},
            "large": {"name": "large", "events_per_topic_per_year": 50, "num_domains": 4},
            "stress": {"name": "stress", "events_per_topic_per_year": 100, "num_domains": 5},
        }

        # Respect BENCHMARK_SCALE env var - can be single scale or comma-separated list
        # Examples: "small", "small,medium", "all"
        requested_scale = os.getenv("BENCHMARK_SCALE", "all").lower().strip()

        if requested_scale == "all":
            self.scale_levels = list(all_scales.values())
        else:
            # Support comma-separated list: "small,medium"
            requested_names = [s.strip() for s in requested_scale.split(",")]
            self.scale_levels = []
            for name in requested_names:
                if name in all_scales:
                    self.scale_levels.append(all_scales[name])
                else:
                    print(f"Warning: Unknown scale '{name}', skipping. Valid: {list(all_scales.keys())}")

            if not self.scale_levels:
                print(f"Error: No valid scales specified. Using 'small' as default.")
                self.scale_levels = [all_scales["small"]]

        print(f"Benchmark will run scales: {[s['name'] for s in self.scale_levels]}")

        # Executor configuration
        # USE_NEW_EXECUTOR=true enables direct use case execution (benchmark purity)
        # USE_NEW_EXECUTOR=false uses HTTP mode (cloud deployments)
        self.use_new_executor = os.getenv("USE_NEW_EXECUTOR", "false").lower() == "true"

        # URLs for both modes
        self.controller_url = os.getenv("CONTROLLER_URL", "http://localhost:8080")
        self.worker_url = os.getenv("PASSIVE_WORKER_URL", "http://localhost:8082")
        self.slm_service_url = os.getenv("SLM_SERVICE_URL", "http://localhost:8081")
        self.slm_timeout = float(os.getenv("SLM_TIMEOUT", "30.0"))

        print(f"Executor mode: {'UseCase (direct)' if self.use_new_executor else 'HTTP'}")
        
    def run_full_benchmark(self):
        """Execute complete benchmark across all scale levels"""
        
        results_summary = {
            "timestamp": time.time(),
            "scale_results": []
        }
        
        for scale_config in self.scale_levels:
            print(f"\n{'='*60}")
            print(f"Running scale: {scale_config['name']}")
            print(f"{'='*60}")
            
            scale_result = self.run_scale_level(scale_config)
            results_summary["scale_results"].append(scale_result)
            
            # Save intermediate results
            self._save_json(results_summary, "benchmark_summary.json")
        
        # Generate final report
        self.generate_research_report(results_summary)
        
        return results_summary
    
    def run_scale_level(self, scale_config: Dict) -> Dict:
        """Run benchmark at specific scale level"""
        
        scale_name = scale_config["name"]
        scale_dir = self.output_dir / scale_name
        scale_dir.mkdir(exist_ok=True)
        
        # Generate dataset
        print(f"Generating dataset for {scale_name}...")
        generator = SyntheticHistoryGeneratorV2(
            events_per_topic_per_year=scale_config["events_per_topic_per_year"],
            perspectives_per_event=3,
            num_domains=scale_config["num_domains"]
        )
        
        dataset = generator.generate_dataset()
        
        # Save dataset artifacts
        self._save_json(dataset, scale_dir / "dataset.json")
        self._save_json(generator.glossary, scale_dir / "glossary.json")
        
        print(f"Generated {len(dataset['events'])} events, {len(dataset['queries'])} queries")
        
        # Run queries against DPR-RC
        print(f"Running DPR-RC queries...")
        dprrc_results = self.run_dprrc_queries(
            dataset['queries'],
            scale_dir / "dprrc_results"
        )
        
        # Run baseline (if local dependencies available)
        print(f"Running baseline queries...")
        baseline_results = self.run_baseline_queries(
            dataset['queries'],
            scale_dir / "baseline_results"
        )
        
        # Evaluate and compare
        comparison = self.compare_results(
            dataset['queries'],
            dprrc_results,
            baseline_results,
            generator.glossary
        )
        
        self._save_json(comparison, scale_dir / "comparison.json")
        
        return {
            "scale": scale_name,
            "config": scale_config,
            "dataset_size": len(dataset['events']),
            "query_count": len(dataset['queries']),
            "dprrc_accuracy": comparison["dprrc_correct_rate"],
            "baseline_accuracy": comparison["baseline_correct_rate"],
            "dprrc_hallucination_rate": comparison["dprrc_hallucination_rate"],
            "baseline_hallucination_rate": comparison["baseline_hallucination_rate"],
            "dprrc_mean_latency_ms": comparison["dprrc_mean_latency"],
            "baseline_mean_latency_ms": comparison["baseline_mean_latency"]
        }
    
    def run_dprrc_queries(self, queries: List[Dict], results_dir: Path) -> List[Dict]:
        """
        Run queries against DPR-RC with full audit trail capture.

        Uses either HTTP mode or UseCase mode based on USE_NEW_EXECUTOR flag.
        Both modes produce identical results, just via different transport layers.
        """
        import asyncio
        from benchmark.infrastructure.executors import create_dprrc_executor

        results_dir.mkdir(exist_ok=True)
        results = []

        # Create executor based on mode
        executor = create_dprrc_executor(
            use_new_executor=self.use_new_executor,
            controller_url=self.controller_url,
            worker_url=self.worker_url,
            slm_url=self.slm_service_url,
            timeout=60.0,
            enable_query_enhancement=True
        )

        print(f"Using executor mode: {executor.execution_mode}")

        for i, query in enumerate(queries):
            query_id = f"query_{i:04d}"
            query_dir = results_dir / query_id
            query_dir.mkdir(exist_ok=True)

            # Save input
            self._save_json({
                "query_text": query["question"],
                "timestamp_context": query.get("timestamp_context"),
                "query_type": query.get("type")
            }, query_dir / "input.json")

            # Save ground truth
            self._save_json({
                "expected_consensus": query.get("expected_consensus", []),
                "expected_disputed": query.get("expected_disputed", []),
                "expected_sources": query.get("expected_sources", [])
            }, query_dir / "ground_truth.json")

            # Execute query via executor (async)
            try:
                result = asyncio.run(executor.execute(
                    query=query["question"],
                    query_id=query_id,
                    timestamp_context=query.get("timestamp_context")
                ))

                if result.success:
                    # Save system output
                    self._save_json({
                        "final_response": result.response,
                        "confidence": result.confidence,
                        "sources": result.metadata.get("sources", []),
                        "status": result.metadata.get("status", "SUCCESS"),
                        "execution_mode": result.metadata.get("execution_mode")
                    }, query_dir / "system_output.json")

                    # Fetch audit trail and exchange history from Cloud Logging (only for HTTP mode)
                    if executor.execution_mode == "http":
                        # Fetch raw audit trail (backwards compatibility)
                        audit_trail = self._fetch_audit_trail(query_id)
                        if audit_trail:
                            self._save_json(audit_trail, query_dir / "audit_trail.json")

                        # Fetch structured exchange history using new download script
                        exchange_history = self._fetch_exchange_history(query_id)
                        if exchange_history:
                            self._save_json(exchange_history, query_dir / "exchange_history.json")

                    results.append({
                        "query_id": query_id,
                        "response": result.response,
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "success": True
                    })
                else:
                    results.append({
                        "query_id": query_id,
                        "response": "",
                        "confidence": 0,
                        "latency_ms": result.latency_ms,
                        "success": False,
                        "error": result.error
                    })

            except Exception as e:
                results.append({
                    "query_id": query_id,
                    "response": "",
                    "confidence": 0,
                    "latency_ms": 0,
                    "success": False,
                    "error": str(e)
                })

        return results
    
    def run_baseline_queries(self, queries: List[Dict], results_dir: Path) -> List[Dict]:
        """Run baseline RAG queries (local fallback if cloud unavailable)"""

        results_dir.mkdir(exist_ok=True)
        results = []

        # Silence tokenizers parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            from dpr_rc.passive_agent import PassiveWorker
            worker = PassiveWorker()

            for i, query in enumerate(queries):
                query_id = f"query_{i:04d}"

                # Convert timestamp_context to shard_id format
                # timestamp_context is like "2015-12-31", shard_id should be "shard_2015"
                timestamp_context = query.get("timestamp_context", "")
                if timestamp_context:
                    year = timestamp_context.split("-")[0]
                    shard_id = f"shard_{year}"
                else:
                    shard_id = "shard_2020"  # default

                start = time.perf_counter()
                doc = worker.retrieve(query["question"], shard_id, timestamp_context)
                latency_ms = (time.perf_counter() - start) * 1000
                
                if doc:
                    results.append({
                        "query_id": query_id,
                        "response": doc["content"],
                        "confidence": 1.0,  # Naive RAG is always confident
                        "latency_ms": latency_ms,
                        "success": True
                    })
                else:
                    results.append({
                        "query_id": query_id,
                        "response": "",
                        "confidence": 0,
                        "latency_ms": latency_ms,
                        "success": False
                    })
        except Exception as e:
            print(f"Baseline execution failed: {e}")
            # Return empty results
            results = [{"query_id": f"query_{i:04d}", "success": False} for i in range(len(queries))]
        
        self._save_json(results, results_dir / "results_baseline.json")
        return results

    def detect_hallucination_via_slm(
        self,
        query: str,
        ground_truth: Dict,
        system_response: str,
        glossary: Dict,
        confidence: float
    ) -> Dict:
        """
        Use SLM to determine if response contains hallucinations.

        This replaces naive string matching with semantic understanding.

        Args:
            query: The original query
            ground_truth: Expected consensus/disputed claims from dataset
            system_response: What A* returned
            glossary: Valid phonotactic terms and their definitions
            confidence: How certain the system was (0-1)

        Returns:
            {
                "has_hallucination": bool,
                "hallucination_type": str or None,
                "explanation": str,
                "severity": str,
                "flagged_content": list
            }
        """
        try:
            # Extract valid terms from glossary to send to SLM
            valid_terms = []

            # Add physics terms
            if 'physics' in glossary:
                valid_terms.extend(list(glossary.get('physics', {}).get('particles', {}).keys())[:20])
                valid_terms.extend(list(glossary.get('physics', {}).get('phenomena', {}).keys())[:20])

            # Add domain terms
            for domain_name, domain_data in glossary.get('domains', {}).items():
                valid_terms.extend(list(domain_data.get('concepts', {}).keys())[:10])

            # Limit to reasonable size (SLM will sample further)
            valid_terms = valid_terms[:50]

            # Call SLM service with retry logic (exponential backoff)
            max_retries = 3
            base_delay = 1.0  # seconds

            last_error = None
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.slm_service_url}/check_hallucination",
                        json={
                            "query": query,
                            "system_response": system_response,
                            "ground_truth": ground_truth,
                            "valid_terms": valid_terms,
                            "confidence": confidence
                        },
                        timeout=self.slm_timeout
                    )

                    if response.status_code == 200:
                        # Parse the HallucinationCheckResponse
                        result = response.json()
                        return {
                            "has_hallucination": result.get("has_hallucination", False),
                            "hallucination_type": result.get("hallucination_type"),
                            "explanation": result.get("explanation", ""),
                            "severity": result.get("severity", "none"),
                            "flagged_content": result.get("flagged_content", [])
                        }
                    elif response.status_code >= 500:
                        # Server error - retry
                        last_error = f"HTTP {response.status_code}"
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"SLM service error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                            time.sleep(delay)
                            continue
                    else:
                        # Client error (4xx) - don't retry
                        print(f"SLM hallucination detection failed: HTTP {response.status_code}")
                        break

                except requests.Timeout:
                    last_error = "timeout"
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"SLM service timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    break
                except requests.ConnectionError as e:
                    last_error = f"connection error: {e}"
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"SLM connection error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    break

            # All retries exhausted, fall back
            print(f"SLM hallucination detection failed after {max_retries} attempts ({last_error})")
            return self._fallback_hallucination_detection(
                system_response, glossary, confidence
            )

        except Exception as e:
            print(f"Error in SLM hallucination detection: {e}")
            return self._fallback_hallucination_detection(
                system_response, glossary, confidence
            )

    def _fallback_hallucination_detection(
        self,
        response: str,
        glossary: Dict,
        confidence: float
    ) -> Dict:
        """
        Improved fallback when SLM is unavailable.
        More sophisticated than pure string matching.
        """
        # Handle None response (raw semantic quadrant mode)
        if response is None:
            return {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "No response text to evaluate (raw semantic quadrant mode)",
                "severity": "none",
                "flagged_content": []
            }

        # Build set of valid terms from glossary
        valid_terms = set()

        # Add physics terms
        if 'physics' in glossary:
            valid_terms.update(glossary['physics'].get('particles', {}).keys())
            valid_terms.update(glossary['physics'].get('phenomena', {}).keys())
            valid_terms.update(glossary['physics'].get('constants', {}).keys())

        # Add domain terms
        for domain_name, domain_data in glossary.get('domains', {}).items():
            valid_terms.update(domain_data.get('concepts', {}).keys())
            valid_terms.add(domain_name)

        # Common English words that should NOT be flagged
        # FIX: Expanded whitelist to prevent false positives on common words
        common_words = {
            # Articles and basic words
            'No', 'Yes', 'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'With',
            'By', 'From', 'As', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being',
            'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Will', 'Would', 'Should',
            'Could', 'May', 'Might', 'Must', 'Can', 'This', 'That', 'These', 'Those',
            # Academic/research terms
            'Research', 'Study', 'Analysis', 'Results', 'Data', 'Findings', 'Progress',
            'Development', 'Breakthrough', 'Discovery', 'Experiment', 'Observation',
            'Review', 'Article', 'Team', 'New', 'Significant', 'Discussion',
            'Implications', 'Focusing', 'Through', 'Challenges', 'Remain',
            # Actions/states
            'Aligns', 'Driven', 'Conclusions', 'Drawn', 'Protocol', 'Predictions',
            'Status', 'Milestone', 'Progress', 'Achieved', 'Improved', 'Showing',
            'Improvement', 'Point', 'Area', 'Metrics', 'Domain'
        }

        # Extract capitalized words (potential proper nouns/terms)
        words = response.split()
        suspicious_terms = []

        for word in words:
            # Clean word (remove punctuation)
            clean_word = word.strip('.,!?;:()"\'')
            if not clean_word:
                continue

            # Check if it's capitalized and not a common word
            if clean_word[0].isupper() and clean_word not in common_words:
                # Check if it's in valid terms
                if clean_word not in valid_terms:
                    suspicious_terms.append(clean_word)

        # If confidence is low or response indicates uncertainty, be lenient
        is_uncertain = confidence < 0.7 or any(
            word in response.lower()
            for word in ['uncertain', 'mixed', 'perspectives', 'disputed', 'conflicting']
        )

        # Only flag as hallucination if:
        # 1. There are suspicious terms
        # 2. System is confident (or if uncertain, terms are egregious)
        if suspicious_terms and (not is_uncertain or len(suspicious_terms) > 5):
            return {
                "has_hallucination": True,
                "hallucination_type": "invalid_term",
                "explanation": f"Found terms not in glossary: {', '.join(suspicious_terms[:5])}",
                "severity": "high" if not is_uncertain else "medium",
                "flagged_content": suspicious_terms
            }
        else:
            return {
                "has_hallucination": False,
                "hallucination_type": None,
                "explanation": "No significant hallucinations detected",
                "severity": "none",
                "flagged_content": []
            }

    def _extract_valid_terms(self, glossary: Dict) -> List[str]:
        """Extract valid terms from glossary for hallucination detection."""
        valid_terms = []

        # Add physics terms
        if 'physics' in glossary:
            valid_terms.extend(list(glossary.get('physics', {}).get('particles', {}).keys())[:20])
            valid_terms.extend(list(glossary.get('physics', {}).get('phenomena', {}).keys())[:20])

        # Add domain terms
        for domain_name, domain_data in glossary.get('domains', {}).items():
            valid_terms.extend(list(domain_data.get('concepts', {}).keys())[:10])

        # Limit to reasonable size (SLM will sample further)
        return valid_terms[:50]

    def batch_detect_hallucination_via_slm(
        self,
        check_requests: List[Dict]
    ) -> List[Dict]:
        """
        Batch hallucination detection - sends multiple requests in one HTTP call.

        Args:
            check_requests: List of dicts with keys:
                - query: str
                - system_response: str
                - ground_truth: dict
                - valid_terms: list[str]
                - confidence: float
                - trace_id: str (optional)

        Returns:
            List of hallucination detection results in same order as requests
        """
        if not check_requests:
            return []

        # If SLM service URL is not configured, fall back to rule-based detection
        if not self.slm_service_url:
            print("No SLM service URL configured, using fallback detection for all requests")
            return [
                self._fallback_hallucination_detection(
                    req["system_response"],
                    {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                    req["confidence"]
                )
                for req in check_requests
            ]

        try:
            # Make batch request to SLM service
            response = requests.post(
                f"{self.slm_service_url}/batch_check_hallucination",
                json=check_requests,
                timeout=self.slm_timeout * 2  # Longer timeout for batch
            )

            if response.status_code == 200:
                batch_response = response.json()
                results = batch_response.get("results", [])

                # Validate response count matches request count
                if len(results) != len(check_requests):
                    print(f"Batch response count mismatch: got {len(results)}, expected {len(check_requests)}")
                    # Fall back to individual detection for all requests
                    return [
                        self._fallback_hallucination_detection(
                            req["system_response"],
                            {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                            req["confidence"]
                        )
                        for req in check_requests
                    ]

                # Convert results to expected format
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "has_hallucination": result.get("has_hallucination", False),
                        "hallucination_type": result.get("hallucination_type"),
                        "explanation": result.get("explanation", ""),
                        "severity": result.get("severity", "none"),
                        "flagged_content": result.get("flagged_content", [])
                    })

                return formatted_results
            else:
                print(f"Batch hallucination detection failed: HTTP {response.status_code}")
                # Fall back to individual detection for all requests
                return [
                    self._fallback_hallucination_detection(
                        req["system_response"],
                        {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                        req["confidence"]
                    )
                    for req in check_requests
                ]

        except Exception as e:
            print(f"Error in batch hallucination detection: {e}")
            # Fall back to individual detection for all requests
            return [
                self._fallback_hallucination_detection(
                    req["system_response"],
                    {"physics": {"particles": {t: {} for t in req["valid_terms"]}}},
                    req["confidence"]
                )
                for req in check_requests
            ]

    def compare_results(
        self,
        queries: List[Dict],
        dprrc_results: List[Dict],
        baseline_results: List[Dict],
        glossary: Dict
    ) -> Dict:
        """
        Compare DPR-RC vs baseline with superposition-aware evaluation.

        Uses batch hallucination detection for efficiency. Requests are queued
        in INTERLEAVED order (dprrc_0, baseline_0, dprrc_1, baseline_1, ...)
        rather than grouped order. This interleaving ensures:
        - Balanced load if batch is split across SLM instances
        - Easier debugging (requests correspond to query iteration order)
        - Natural result ordering for sequential processing

        Args:
            queries: List of query dicts with question, expected_consensus, expected_disputed
            dprrc_results: List of DPR-RC query results
            baseline_results: List of baseline query results
            glossary: Valid phonotactic terms for hallucination detection

        Returns:
            Dict with correctness rates, hallucination counts, latency stats
        """

        dprrc_correct = 0
        baseline_correct = 0
        dprrc_hallucinations = []
        baseline_hallucinations = []
        dprrc_latencies = []
        baseline_latencies = []

        # Batch hallucination detection requests
        hallucination_check_requests = []
        request_metadata = []  # Track which requests are for dprrc vs baseline

        for i, query in enumerate(queries):
            dprrc = dprrc_results[i] if i < len(dprrc_results) else {"success": False}
            baseline = baseline_results[i] if i < len(baseline_results) else {"success": False}

            # Extract expected entities from query using EvaluationService
            question_entities = EvaluationService.extract_entities_from_question(
                query["question"]
            )

            ground_truth = {
                "expected_consensus": query.get("expected_consensus", []),
                "expected_disputed": query.get("expected_disputed", [])
            }

            # Evaluate DPR-RC (superposition-aware)
            if dprrc.get("success"):
                response = dprrc.get("response", "")
                confidence = dprrc.get("confidence", 0)

                # Use EvaluationService for correctness check
                correctness_result = EvaluationService.evaluate_superposition_correctness(
                    response=response,
                    expected_entities=question_entities,
                    recall_threshold=0.5,
                    min_response_length=10
                )

                if correctness_result.is_correct:
                    dprrc_correct += 1

                # Queue hallucination check for batch processing
                hallucination_check_requests.append({
                    "query": query.get("question", ""),
                    "system_response": response,
                    "ground_truth": ground_truth,
                    "valid_terms": self._extract_valid_terms(glossary),
                    "confidence": confidence,
                    "trace_id": dprrc.get("query_id", f"dprrc_{i}")
                })
                request_metadata.append({
                    "type": "dprrc",
                    "index": i,
                    "query_id": dprrc.get("query_id"),
                    "confidence": confidence
                })

                dprrc_latencies.append(dprrc.get("latency_ms", 0))

            # Evaluate baseline (not superposition-aware)
            if baseline.get("success"):
                response = baseline.get("response", "")

                # Use EvaluationService for correctness check
                correctness_result = EvaluationService.evaluate_correctness(
                    response=response,
                    expected_entities=question_entities,
                    recall_threshold=0.5,
                    min_response_length=20
                )

                if correctness_result.is_correct:
                    baseline_correct += 1

                # Queue hallucination check for batch processing
                hallucination_check_requests.append({
                    "query": query.get("question", ""),
                    "system_response": response,
                    "ground_truth": ground_truth,
                    "valid_terms": self._extract_valid_terms(glossary),
                    "confidence": 1.0,  # Baseline is always confident
                    "trace_id": baseline.get("query_id", f"baseline_{i}")
                })
                request_metadata.append({
                    "type": "baseline",
                    "index": i,
                    "query_id": baseline.get("query_id")
                })

                baseline_latencies.append(baseline.get("latency_ms", 0))

        # Batch process all hallucination checks (I/O stays in this method)
        if hallucination_check_requests:
            batch_results = self.batch_detect_hallucination_via_slm(hallucination_check_requests)

            # Process batch results and map back to dprrc/baseline
            for result, metadata in zip(batch_results, request_metadata):
                if result["has_hallucination"]:
                    hallucination_entry = {
                        "query_id": metadata["query_id"],
                        "type": result["hallucination_type"],
                        "severity": result["severity"],
                        "explanation": result["explanation"],
                        "flagged_content": result["flagged_content"]
                    }

                    if metadata["type"] == "dprrc":
                        hallucination_entry["confidence"] = metadata["confidence"]
                        dprrc_hallucinations.append(hallucination_entry)
                    else:  # baseline
                        baseline_hallucinations.append(hallucination_entry)

        total_queries = len(queries)

        return {
            "total_queries": total_queries,
            "dprrc_correct_count": dprrc_correct,
            "baseline_correct_count": baseline_correct,
            "dprrc_correct_rate": dprrc_correct / total_queries if total_queries > 0 else 0,
            "baseline_correct_rate": baseline_correct / total_queries if total_queries > 0 else 0,
            "dprrc_hallucination_count": len(dprrc_hallucinations),
            "baseline_hallucination_count": len(baseline_hallucinations),
            "dprrc_hallucination_rate": len(dprrc_hallucinations) / total_queries if total_queries > 0 else 0,
            "baseline_hallucination_rate": len(baseline_hallucinations) / total_queries if total_queries > 0 else 0,
            "dprrc_mean_latency": np.mean(dprrc_latencies) if dprrc_latencies else 0,
            "baseline_mean_latency": np.mean(baseline_latencies) if baseline_latencies else 0,
            "dprrc_p95_latency": np.percentile(dprrc_latencies, 95) if dprrc_latencies else 0,
            "baseline_p95_latency": np.percentile(baseline_latencies, 95) if baseline_latencies else 0,
            "dprrc_hallucination_details": dprrc_hallucinations,
            "baseline_hallucination_details": baseline_hallucinations
        }
    
    def generate_research_report(self, results_summary: Dict):
        """Generate publication-ready markdown report"""
        
        report_path = self.output_dir / "RESEARCH_REPORT.md"
        
        with open(report_path, "w") as f:
            f.write("# DPR-RC Research Benchmark Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents empirical evaluation of Distributed Passive Retrieval ")
            f.write("with Resonant Consensus (DPR-RC) against baseline RAG architectures.\n\n")
            
            f.write("## Methodology\n\n")
            f.write("- **Dataset**: Phonotactic synthetic history (zero real-world term overlap)\n")
            f.write("- **Evaluation**: Superposition-aware correctness (correct answer present)\n")
            f.write("- **Hallucination Detection**: Terms not in generated glossary\n")
            f.write("- **Scale Levels**: Progressive scaling to find failure thresholds\n\n")
            
            f.write("## Results by Scale\n\n")
            f.write("| Scale | Events | Queries | DPR-RC Accuracy | Baseline Accuracy | DPR-RC Halluc. | Baseline Halluc. |\n")
            f.write("|-------|--------|---------|-----------------|-------------------|----------------|------------------|\n")
            
            for result in results_summary["scale_results"]:
                f.write(f"| {result['scale']} | ")
                f.write(f"{result['dataset_size']} | ")
                f.write(f"{result['query_count']} | ")
                f.write(f"{result['dprrc_accuracy']:.2%} | ")
                f.write(f"{result['baseline_accuracy']:.2%} | ")
                f.write(f"{result['dprrc_hallucination_rate']:.2%} | ")
                f.write(f"{result['baseline_hallucination_rate']:.2%} |\n")
            
            f.write("\n## Latency Comparison\n\n")
            f.write("| Scale | DPR-RC Mean (ms) | Baseline Mean (ms) | Overhead |\n")
            f.write("|-------|------------------|--------------------|-----------|\n")
            
            for result in results_summary["scale_results"]:
                dprrc_lat = result['dprrc_mean_latency_ms']
                baseline_lat = result['baseline_mean_latency_ms']
                overhead = ((dprrc_lat / baseline_lat) - 1) * 100 if baseline_lat > 0 else 0
                
                f.write(f"| {result['scale']} | ")
                f.write(f"{dprrc_lat:.1f} | ")
                f.write(f"{baseline_lat:.1f} | ")
                f.write(f"+{overhead:.1f}% |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Find hallucination threshold
            for i, result in enumerate(results_summary["scale_results"]):
                if result['baseline_hallucination_rate'] > 0.1 and result['dprrc_hallucination_rate'] < 0.05:
                    f.write(f"**Hallucination Threshold**: At {result['scale']} scale ")
                    f.write(f"({result['dataset_size']} events), baseline hallucination rate ")
                    f.write(f"reaches {result['baseline_hallucination_rate']:.1%} while DPR-RC ")
                    f.write(f"maintains {result['dprrc_hallucination_rate']:.1%}.\n\n")
                    break
            
            f.write("## Reproducibility\n\n")
            f.write("All artifacts for peer review are available in:\n")
            f.write(f"- `{self.output_dir}/`\n")
            f.write("  - `{scale}/dataset.json` - Generated datasets\n")
            f.write("  - `{scale}/glossary.json` - Phonotactic term definitions\n")
            f.write("  - `{scale}/dprrc_results/query_XXXX/` - Per-query audit trails\n")
            f.write("  - `{scale}/comparison.json` - Detailed evaluation metrics\n\n")
            
            f.write("## Audit Trail Structure\n\n")
            f.write("Each DPR-RC query includes:\n")
            f.write("- `input.json` - Query text and context\n")
            f.write("- `ground_truth.json` - Expected consensus/disputed claims\n")
            f.write("- `system_output.json` - Final response and confidence\n")
            f.write("- `audit_trail.json` - Complete L1/L2/L3 execution trace\n\n")
        
        print(f"\n✓ Research report generated: {report_path}")
    
    def _fetch_audit_trail(self, trace_id: str) -> Dict:
        """Fetch complete audit trail from Cloud Logging (raw logs)"""
        try:
            import subprocess
            result = subprocess.run([
                "gcloud", "logging", "read",
                f'jsonPayload.trace_id="{trace_id}"',
                "--limit", "100",
                "--format", "json"
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return json.loads(result.stdout)
        except:
            pass

        return {}

    def _fetch_exchange_history(self, trace_id: str) -> Dict:
        """
        Fetch structured exchange history using download_query_history script.

        Returns complete message exchange sequence with proper chronological
        ordering and message type categorization.
        """
        try:
            import subprocess
            import sys

            # Use the download_query_history script
            script_path = Path(__file__).parent.parent / "scripts" / "download_query_history.py"
            if not script_path.exists():
                print(f"Warning: Exchange history script not found at {script_path}")
                return {}

            # Run script with JSON output format
            result = subprocess.run([
                sys.executable, str(script_path),
                trace_id,
                "--format", "json"
            ], capture_output=True, text=True, timeout=15)

            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
            elif result.stderr:
                print(f"Warning: Error fetching exchange history for {trace_id}: {result.stderr[:200]}")
        except Exception as e:
            print(f"Warning: Failed to fetch exchange history for {trace_id}: {e}")

        return {}
    
    def _save_json(self, data: Any, path: Path):
        """Save JSON with pretty printing"""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def upload_results_to_gcs(local_dir: Path):
    """Upload benchmark results to GCS for retrieval."""
    bucket_name = os.getenv("HISTORY_BUCKET")
    if not bucket_name:
        print("HISTORY_BUCKET not set, skipping GCS upload")
        return

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Count and upload all files in results directory
        files_to_upload = list(local_dir.rglob("*"))
        files_to_upload = [f for f in files_to_upload if f.is_file()]
        print(f"Found {len(files_to_upload)} files to upload from {local_dir}")

        if not files_to_upload:
            print("Warning: No files found to upload!")
            return

        uploaded_count = 0
        for file_path in files_to_upload:
            relative_path = file_path.relative_to(local_dir)
            blob_name = f"benchmark_results/{relative_path}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            uploaded_count += 1
            # Only print every 10th file to reduce noise
            if uploaded_count <= 5 or uploaded_count % 10 == 0:
                print(f"  [{uploaded_count}/{len(files_to_upload)}] gs://{bucket_name}/{blob_name}")

        print(f"✓ Uploaded {uploaded_count} files to gs://{bucket_name}/benchmark_results/")
    except Exception as e:
        print(f"ERROR: Could not upload results to GCS: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run complete research benchmark"""
    # Silence HuggingFace tokenizers parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    suite = ResearchBenchmarkSuite()
    results = None

    try:
        results = suite.run_full_benchmark()

        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        print(f"Results directory: {suite.output_dir}")
        print(f"Research report: {suite.output_dir}/RESEARCH_REPORT.md")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"BENCHMARK FAILED: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
    finally:
        # Always try to upload results to GCS, even on failure
        # This ensures partial results are available for debugging
        if os.getenv("HISTORY_BUCKET"):
            print("\n--- Uploading results to GCS ---")
            print(f"Bucket: gs://{os.getenv('HISTORY_BUCKET')}/benchmark_results/")
            upload_results_to_gcs(suite.output_dir)
        else:
            print("\nHISTORY_BUCKET not set, skipping GCS upload")


if __name__ == "__main__":
    main()
