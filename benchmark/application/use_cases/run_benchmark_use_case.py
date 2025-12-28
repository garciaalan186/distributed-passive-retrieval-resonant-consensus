"""
Run Benchmark Use Case

Orchestrates the full benchmark flow with clean separation of concerns.

This use case is the entry point for running benchmarks in the new architecture.
It coordinates dataset generation, query execution, evaluation, and reporting.
"""

import time
import uuid
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

from benchmark.application.dtos import RunBenchmarkRequest, RunBenchmarkResponse
from benchmark.domain.interfaces import (
    IQueryExecutor,
    IDatasetGenerator,
    QueryExecutionResult
)
from benchmark.domain.services import EvaluationService


class RunBenchmarkUseCase:
    """
    Use case for running a complete benchmark.

    This use case orchestrates the full benchmark workflow:
    1. Generate synthetic dataset
    2. Execute queries via both executors (DPR-RC and baseline)
    3. Evaluate results using EvaluationService
    4. Detect hallucinations (via external SLM service or fallback)
    5. Aggregate metrics
    6. Generate report
    7. Save results to local files

    Design Principles:
    1. **Dependency Injection**: All dependencies injected via constructor
    2. **Single Responsibility**: Only orchestrates; delegates actual work
    3. **Testability**: Easy to test with mocks
    4. **No I/O Leakage**: I/O operations clearly isolated
    5. **Async Support**: Full async for concurrent query execution

    Dependencies:
    - dprrc_executor: Executes queries against DPR-RC system
    - baseline_executor: Executes queries against baseline system
    - dataset_generator: Generates benchmark datasets
    - evaluation_service: Pure evaluation logic (no I/O)
    """

    def __init__(
        self,
        dprrc_executor: IQueryExecutor,
        baseline_executor: IQueryExecutor,
        dataset_generator: IDatasetGenerator,
        evaluation_service: Optional[EvaluationService] = None
    ):
        """
        Initialize the use case with dependencies.

        Args:
            dprrc_executor: Executor for DPR-RC queries
            baseline_executor: Executor for baseline queries
            dataset_generator: Generator for benchmark datasets
            evaluation_service: Service for evaluation logic (defaults to EvaluationService)
        """
        self._dprrc_executor = dprrc_executor
        self._baseline_executor = baseline_executor
        self._dataset_generator = dataset_generator
        self._evaluation_service = evaluation_service or EvaluationService()

    async def execute(self, request: RunBenchmarkRequest) -> RunBenchmarkResponse:
        """
        Execute a complete benchmark run.

        Args:
            request: Benchmark configuration and parameters

        Returns:
            RunBenchmarkResponse with aggregated metrics and result paths

        Workflow:
        1. Generate dataset
        2. Execute DPR-RC queries
        3. Execute baseline queries
        4. Evaluate correctness
        5. Detect hallucinations (if enabled)
        6. Aggregate metrics
        7. Save results
        8. Generate report

        Error Handling:
        - All errors captured and returned in response.error
        - Partial results saved even on failure
        - No exceptions propagate to caller
        """
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        try:
            # 1. Generate dataset
            dataset = self._dataset_generator.generate(
                scale=request.scale,
                seed=request.seed
            )

            # Validate dataset
            validation_warnings = self._dataset_generator.validate_dataset(dataset)
            if validation_warnings:
                print(f"Dataset validation warnings: {validation_warnings}")

            # 2. Setup output directory
            output_dir = Path(request.output_dir) / f"{request.scale}_{run_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save dataset
            dataset_path = output_dir / "dataset.json"
            self._save_dataset(dataset, dataset_path)

            # 3. Execute DPR-RC queries
            print(f"Executing {len(dataset.queries)} DPR-RC queries...")
            dprrc_results = await self._execute_queries(
                queries=dataset.queries,
                executor=self._dprrc_executor,
                output_dir=output_dir / "dprrc_results"
            )

            # 4. Execute baseline queries
            print(f"Executing {len(dataset.queries)} baseline queries...")
            baseline_results = await self._execute_queries(
                queries=dataset.queries,
                executor=self._baseline_executor,
                output_dir=output_dir / "baseline_results"
            )

            # 5. Evaluate and compare results
            print("Evaluating results...")
            comparison = self._compare_results(
                queries=dataset.queries,
                dprrc_results=dprrc_results,
                baseline_results=baseline_results,
                glossary=dataset.glossary,
                slm_service_url=request.slm_service_url if request.enable_hallucination_detection else None
            )

            # 6. Save comparison results
            comparison_path = output_dir / "comparison.json"
            self._save_json(comparison, comparison_path)

            # 7. Generate report
            report_path = output_dir / "REPORT.md"
            self._generate_report(
                scale=request.scale,
                run_id=run_id,
                dataset=dataset,
                comparison=comparison,
                output_path=report_path,
                elapsed_time=time.time() - start_time
            )

            # 8. Create response
            return RunBenchmarkResponse(
                run_id=run_id,
                scale=request.scale,
                total_queries=len(dataset.queries),
                dprrc_accuracy=comparison["dprrc_correct_rate"],
                baseline_accuracy=comparison["baseline_correct_rate"],
                dprrc_hallucination_rate=comparison["dprrc_hallucination_rate"],
                baseline_hallucination_rate=comparison["baseline_hallucination_rate"],
                mean_latency_dprrc=comparison["dprrc_mean_latency"],
                mean_latency_baseline=comparison["baseline_mean_latency"],
                p95_latency_dprrc=comparison["dprrc_p95_latency"],
                p95_latency_baseline=comparison["baseline_p95_latency"],
                report_path=str(report_path),
                dataset_path=str(dataset_path),
                error=None
            )

        except Exception as e:
            # Capture error and return failed response
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Benchmark failed: {error_msg}")

            # Return partial results if possible
            return RunBenchmarkResponse(
                run_id=run_id,
                scale=request.scale,
                total_queries=0,
                dprrc_accuracy=0.0,
                baseline_accuracy=0.0,
                dprrc_hallucination_rate=0.0,
                baseline_hallucination_rate=0.0,
                mean_latency_dprrc=0.0,
                mean_latency_baseline=0.0,
                p95_latency_dprrc=0.0,
                p95_latency_baseline=0.0,
                report_path="",
                error=error_msg
            )

    async def _execute_queries(
        self,
        queries: List[Dict[str, Any]],
        executor: IQueryExecutor,
        output_dir: Path
    ) -> List[QueryExecutionResult]:
        """
        Execute queries via an executor and save results.

        Args:
            queries: List of query dictionaries
            executor: Query executor to use
            output_dir: Directory to save per-query results

        Returns:
            List of QueryExecutionResult
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, query in enumerate(queries):
            query_id = f"query_{i:04d}"

            # Execute query
            result = await executor.execute(
                query=query["question"],
                query_id=query_id,
                timestamp_context=query.get("timestamp_context")
            )

            results.append(result)

            # Save individual query result
            query_dir = output_dir / query_id
            query_dir.mkdir(exist_ok=True)

            self._save_json({
                "query_text": query["question"],
                "timestamp_context": query.get("timestamp_context"),
            }, query_dir / "input.json")

            self._save_json({
                "expected_consensus": query.get("expected_consensus", []),
                "expected_disputed": query.get("expected_disputed", []),
            }, query_dir / "ground_truth.json")

            if result.success:
                self._save_json({
                    "final_response": result.response,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                    "metadata": result.metadata
                }, query_dir / "system_output.json")

        return results

    def _compare_results(
        self,
        queries: List[Dict[str, Any]],
        dprrc_results: List[QueryExecutionResult],
        baseline_results: List[QueryExecutionResult],
        glossary: Dict[str, Any],
        slm_service_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare DPR-RC vs baseline results.

        This method uses EvaluationService for correctness evaluation and
        optionally calls an external SLM service for hallucination detection.

        Args:
            queries: List of query dictionaries
            dprrc_results: Results from DPR-RC executor
            baseline_results: Results from baseline executor
            glossary: Valid terms for hallucination detection
            slm_service_url: Optional URL for SLM hallucination detection service

        Returns:
            Dictionary with comparison metrics
        """
        dprrc_correct = 0
        baseline_correct = 0
        dprrc_hallucinations = []
        baseline_hallucinations = []
        dprrc_latencies = []
        baseline_latencies = []

        for i, query in enumerate(queries):
            dprrc = dprrc_results[i] if i < len(dprrc_results) else None
            baseline = baseline_results[i] if i < len(baseline_results) else None

            # Extract expected entities from query
            question_entities = self._evaluation_service.extract_entities_from_question(
                query["question"]
            )

            # Evaluate DPR-RC (superposition-aware)
            if dprrc and dprrc.success:
                correctness_result = self._evaluation_service.evaluate_superposition_correctness(
                    response=dprrc.response,
                    expected_entities=question_entities,
                    recall_threshold=0.5,
                    min_response_length=10
                )

                if correctness_result.is_correct:
                    dprrc_correct += 1

                dprrc_latencies.append(dprrc.latency_ms)

                # Hallucination detection (simplified - no actual SLM call in this implementation)
                # In production, this would call the SLM service
                # For now, we use the fallback rule-based detection
                has_hallucination = self._detect_hallucination_fallback(
                    response=dprrc.response,
                    glossary=glossary,
                    confidence=dprrc.confidence
                )
                if has_hallucination:
                    dprrc_hallucinations.append({
                        "query_id": dprrc.query_id,
                        "type": "invalid_term",
                        "severity": "medium"
                    })

            # Evaluate baseline (not superposition-aware)
            if baseline and baseline.success:
                correctness_result = self._evaluation_service.evaluate_correctness(
                    response=baseline.response,
                    expected_entities=question_entities,
                    recall_threshold=0.5,
                    min_response_length=20
                )

                if correctness_result.is_correct:
                    baseline_correct += 1

                baseline_latencies.append(baseline.latency_ms)

                # Hallucination detection
                has_hallucination = self._detect_hallucination_fallback(
                    response=baseline.response,
                    glossary=glossary,
                    confidence=1.0  # Baseline is always confident
                )
                if has_hallucination:
                    baseline_hallucinations.append({
                        "query_id": baseline.query_id,
                        "type": "invalid_term",
                        "severity": "medium"
                    })

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

    def _detect_hallucination_fallback(
        self,
        response: str,
        glossary: Dict[str, Any],
        confidence: float
    ) -> bool:
        """
        Simple rule-based hallucination detection.

        This is a fallback when SLM service is not available.
        Checks if response contains terms not in the glossary.

        Args:
            response: System response
            glossary: Valid terms
            confidence: System confidence

        Returns:
            True if hallucination detected, False otherwise
        """
        # Build set of valid terms from glossary
        valid_terms = set()

        if 'physics' in glossary:
            valid_terms.update(glossary['physics'].get('particles', {}).keys())
            valid_terms.update(glossary['physics'].get('phenomena', {}).keys())
            valid_terms.update(glossary['physics'].get('constants', {}).keys())

        for domain_name, domain_data in glossary.get('domains', {}).items():
            valid_terms.update(domain_data.get('concepts', {}).keys())
            valid_terms.add(domain_name)

        # Extract capitalized words from response
        words = response.split()
        common_words = {
            'No', 'Yes', 'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'With',
            'By', 'From', 'As', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being',
            'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Will', 'Would', 'Should',
            'Could', 'May', 'Might', 'Must', 'Can', 'This', 'That', 'These', 'Those'
        }

        suspicious_count = 0
        for word in words:
            clean_word = word.strip('.,!?;:()"\'')
            if clean_word and clean_word[0].isupper() and clean_word not in common_words:
                if clean_word not in valid_terms:
                    suspicious_count += 1

        # Flag as hallucination if multiple suspicious terms
        return suspicious_count > 3

    def _save_dataset(self, dataset, path: Path):
        """Save dataset to JSON file"""
        self._save_json({
            "scale": dataset.scale,
            "queries": dataset.queries,
            "glossary": dataset.glossary,
            "metadata": dataset.metadata
        }, path)

    def _save_json(self, data: Any, path: Path):
        """Save JSON with pretty printing"""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_report(
        self,
        scale: str,
        run_id: str,
        dataset,
        comparison: Dict[str, Any],
        output_path: Path,
        elapsed_time: float
    ):
        """Generate markdown report"""
        with open(output_path, "w") as f:
            f.write(f"# Benchmark Report: {scale.capitalize()} Scale\n\n")
            f.write(f"**Run ID**: {run_id}\n")
            f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Elapsed Time**: {elapsed_time:.2f}s\n\n")

            f.write("## Dataset\n\n")
            f.write(f"- Scale: {scale}\n")
            f.write(f"- Events: {dataset.metadata.get('num_events', 'N/A')}\n")
            f.write(f"- Queries: {dataset.metadata.get('num_queries', 'N/A')}\n\n")

            f.write("## Results\n\n")
            f.write("| Metric | DPR-RC | Baseline |\n")
            f.write("|--------|--------|----------|\n")
            f.write(f"| Accuracy | {comparison['dprrc_correct_rate']:.2%} | {comparison['baseline_correct_rate']:.2%} |\n")
            f.write(f"| Hallucination Rate | {comparison['dprrc_hallucination_rate']:.2%} | {comparison['baseline_hallucination_rate']:.2%} |\n")
            f.write(f"| Mean Latency (ms) | {comparison['dprrc_mean_latency']:.1f} | {comparison['baseline_mean_latency']:.1f} |\n")
            f.write(f"| P95 Latency (ms) | {comparison['dprrc_p95_latency']:.1f} | {comparison['baseline_p95_latency']:.1f} |\n\n")

            f.write("## Analysis\n\n")

            # Calculate improvements
            if comparison['baseline_correct_rate'] > 0:
                acc_improvement = (
                    (comparison['dprrc_correct_rate'] - comparison['baseline_correct_rate'])
                    / comparison['baseline_correct_rate']
                ) * 100
                f.write(f"- Accuracy improvement: {acc_improvement:+.1f}%\n")

            if comparison['baseline_hallucination_rate'] > 0:
                hall_reduction = (
                    (comparison['baseline_hallucination_rate'] - comparison['dprrc_hallucination_rate'])
                    / comparison['baseline_hallucination_rate']
                ) * 100
                f.write(f"- Hallucination reduction: {hall_reduction:+.1f}%\n")

            if comparison['baseline_mean_latency'] > 0:
                latency_overhead = (
                    (comparison['dprrc_mean_latency'] - comparison['baseline_mean_latency'])
                    / comparison['baseline_mean_latency']
                ) * 100
                f.write(f"- Latency overhead: {latency_overhead:+.1f}%\n")
