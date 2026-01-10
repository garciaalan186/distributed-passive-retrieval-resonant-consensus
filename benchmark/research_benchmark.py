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
from pathlib import Path

from benchmark.synthetic import SyntheticHistoryGeneratorV2
from benchmark.domain.services import EvaluationService
from benchmark.core import HallucinationDetector, ReportGenerator
from benchmark.core.models import SuperpositionEvaluation, HallucinationAnalysis, ResourceMetrics
from benchmark.config import get_config, get_all_scales
from dpr_rc.infrastructure.passive_agent.repositories import ChromaDBRepository


class ResearchBenchmarkSuite:
    """Publication-ready benchmark with complete audit trails"""

    def __init__(self, output_dir: str = "benchmark_results_research"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load configuration
        self._config = get_config()
        self._all_scales = get_all_scales()

        # Parse scale levels from environment
        self.scale_levels = self._parse_scale_levels()

        # Executor configuration
        self.use_new_executor = os.getenv("USE_NEW_EXECUTOR", "false").lower() == "true"

        # Service URLs (env vars override config)
        services_config = self._config.get('services', {})
        self.controller_url = os.getenv("CONTROLLER_URL", services_config.get('controller_url', "http://localhost:8080"))
        self.worker_url = os.getenv("PASSIVE_WORKER_URL", services_config.get('passive_worker_url', "http://localhost:8082"))
        self.slm_service_url = os.getenv("SLM_SERVICE_URL", services_config.get('slm_service_url', "http://localhost:8081"))
        self.slm_timeout = float(os.getenv("SLM_TIMEOUT", str(services_config.get('timeouts', {}).get('slm', 30.0))))

        # Initialize components
        self.hallucination_detector = HallucinationDetector(
            slm_service_url=self.slm_service_url,
            slm_timeout=self.slm_timeout
        )
        self.report_generator = ReportGenerator(self.output_dir)

        print(f"Benchmark will run scales: {[s['name'] for s in self.scale_levels]}")
        print(f"Executor mode: {'UseCase (direct)' if self.use_new_executor else 'HTTP'}")

    def _parse_scale_levels(self) -> List[Dict]:
        """Parse scale levels from BENCHMARK_SCALE env var."""
        requested_scale = os.getenv("BENCHMARK_SCALE", "all").lower().strip()

        if requested_scale == "all":
            return list(self._all_scales.values())

        requested_names = [s.strip() for s in requested_scale.split(",")]
        scale_levels = []
        for name in requested_names:
            if name in self._all_scales:
                scale_levels.append(self._all_scales[name])
            else:
                print(f"Warning: Unknown scale '{name}', skipping. Valid: {list(self._all_scales.keys())}")

        if not scale_levels:
            print(f"Error: No valid scales specified. Using 'small' as default.")
            scale_levels = [self._all_scales["small"]]

        return scale_levels

    def _ingest_dataset(self, dataset: Dict):
        """Ingest dataset into ChromaDB for Direct/UseCase mode."""
        print("\nIngesting dataset into ChromaDB...")

        repo = ChromaDBRepository()

        shards = {}
        for event in dataset.get("events", []):
            try:
                year = event["timestamp"][:4]
                shard_id = f"shard_{year}"

                if shard_id not in shards:
                    shards[shard_id] = []

                doc = {
                    "id": event["id"],
                    "content": event["content"],
                    "metadata": {
                        "id": event["id"],
                        "doc_id": event["id"],
                        "year": int(year),
                        "timestamp": event["timestamp"],
                        "domain": event.get("topic", "general"),
                        "type": "event"
                    }
                }
                shards[shard_id].append(doc)
            except Exception as e:
                print(f"Warning: Skipping event {event.get('id')}: {e}")

        total_inserted = 0
        for shard_id, docs in shards.items():
            try:
                count = repo.bulk_insert(shard_id, docs)
                total_inserted += count
                if count > 0:
                    print(f"  {shard_id}: Inserted {count} documents (skipped {len(docs) - count} duplicates)")
            except Exception as e:
                print(f"Error inserting shard {shard_id}: {e}")

        print(f"Ingestion complete. Total new documents: {total_inserted}")

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

            self._save_json(results_summary, "benchmark_summary.json")

        self.report_generator.generate(results_summary)

        return results_summary

    def run_scale_level(self, scale_config: Dict) -> Dict:
        """Run benchmark at specific scale level"""
        scale_name = scale_config["name"]
        scale_dir = self.output_dir / scale_name
        scale_dir.mkdir(exist_ok=True)

        print(f"Generating dataset for {scale_name}...")
        generator = SyntheticHistoryGeneratorV2(
            events_per_topic_per_year=scale_config["events_per_topic_per_year"],
            perspectives_per_event=3,
            num_domains=scale_config["num_domains"]
        )

        dataset = generator.generate_dataset()

        self._save_json(dataset, scale_dir / "dataset.json")
        self._save_json(generator.glossary, scale_dir / "glossary.json")

        print(f"Generated {len(dataset['events'])} events, {len(dataset['queries'])} queries")

        if self.use_new_executor:
            self._ingest_dataset(dataset)

        print(f"Running DPR-RC queries...")
        dprrc_results = self.run_dprrc_queries(
            dataset['queries'],
            scale_dir / "dprrc_results"
        )

        print(f"Running baseline queries...")
        baseline_results = self.run_baseline_queries(
            dataset['queries'],
            scale_dir / "baseline_results"
        )

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
        """Run queries against DPR-RC with full audit trail capture."""
        import asyncio
        from benchmark.infrastructure.executors import create_dprrc_executor

        results_dir.mkdir(exist_ok=True)

        executor = create_dprrc_executor(
            use_new_executor=self.use_new_executor,
            controller_url=self.controller_url,
            worker_url=self.worker_url,
            slm_url=self.slm_service_url,
            timeout=60.0,
            enable_query_enhancement=True
        )

        print(f"Using executor mode: {executor.execution_mode}")

        enable_parallel = os.getenv("ENABLE_PARALLEL_QUERIES", "true").lower() == "true"
        max_concurrent = int(os.getenv("MAX_CONCURRENT_QUERIES", "6"))

        if not enable_parallel:
            print("Parallel queries disabled, using sequential execution")
            return self._run_queries_sequential(queries, results_dir, executor)

        print(f"Parallel queries enabled (max {max_concurrent} concurrent)")
        return asyncio.run(self._run_queries_parallel(
            queries, results_dir, executor, max_concurrent
        ))

    def _run_queries_sequential(
        self,
        queries: List[Dict],
        results_dir: Path,
        executor
    ) -> List[Dict]:
        """Sequential query execution"""
        import asyncio

        results = []
        for i, query in enumerate(queries):
            query_id = f"query_{i:04d}"
            query_dir = results_dir / query_id
            query_dir.mkdir(exist_ok=True)

            self._save_query_metadata(query, query_dir)

            try:
                result = asyncio.run(executor.execute(
                    query=query["question"],
                    query_id=query_id,
                    timestamp_context=query.get("timestamp_context")
                ))

                results.append(
                    self._process_query_result(result, query_id, query_dir, executor)
                )

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

    async def _run_queries_parallel(
        self,
        queries: List[Dict],
        results_dir: Path,
        executor,
        max_concurrent: int
    ) -> List[Dict]:
        """Parallel query execution with concurrency limit"""
        import asyncio

        results = []

        for i, query in enumerate(queries):
            query_id = f"query_{i:04d}"
            query_dir = results_dir / query_id
            query_dir.mkdir(exist_ok=True)
            self._save_query_metadata(query, query_dir)

        for batch_start in range(0, len(queries), max_concurrent):
            batch_end = min(batch_start + max_concurrent, len(queries))
            batch = queries[batch_start:batch_end]

            print(f"Processing batch {batch_start//max_concurrent + 1}: queries {batch_start}-{batch_end-1}")

            # Include timestamp_context per query (as third element in tuple)
            batch_queries = [
                (f"query_{batch_start + i:04d}", q["question"], q.get("timestamp_context"))
                for i, q in enumerate(batch)
            ]

            try:
                batch_results = await executor.execute_batch(batch_queries)

                for i, result in enumerate(batch_results):
                    query_idx = batch_start + i
                    query_id = f"query_{query_idx:04d}"
                    query_dir = results_dir / query_id

                    results.append(
                        self._process_query_result(result, query_id, query_dir, executor)
                    )

            except Exception as e:
                for i in range(len(batch)):
                    query_idx = batch_start + i
                    results.append({
                        "query_id": f"query_{query_idx:04d}",
                        "response": "",
                        "confidence": 0,
                        "latency_ms": 0,
                        "success": False,
                        "error": f"Batch execution failed: {str(e)}"
                    })

        return results

    def _save_query_metadata(self, query: Dict, query_dir: Path):
        """Save query input and ground truth metadata"""
        self._save_json({
            "query_text": query["question"],
            "timestamp_context": query.get("timestamp_context"),
            "query_type": query.get("type")
        }, query_dir / "input.json")

        self._save_json({
            "expected_consensus": query.get("expected_consensus", []),
            "expected_disputed": query.get("expected_disputed", []),
            "expected_sources": query.get("expected_sources", [])
        }, query_dir / "ground_truth.json")

    def _process_query_result(
        self,
        result,
        query_id: str,
        query_dir: Path,
        executor
    ) -> Dict:
        """Process and save query result"""
        if result.success:
            self._save_json({
                "final_response": result.response,
                "confidence": result.confidence,
                "sources": result.metadata.get("sources", []),
                "status": result.metadata.get("status", "SUCCESS"),
                "execution_mode": result.metadata.get("execution_mode"),
                "superposition": result.metadata.get("superposition")
            }, query_dir / "system_output.json")

            if executor.execution_mode == "http":
                audit_trail = self._fetch_audit_trail(query_id)
                if audit_trail:
                    self._save_json(audit_trail, query_dir / "audit_trail.json")

                exchange_history = self._fetch_exchange_history(query_id)
                if exchange_history:
                    self._save_json(exchange_history, query_dir / "exchange_history.json")

            return {
                "query_id": query_id,
                "response": result.response,
                "confidence": result.confidence,
                "superposition": result.metadata.get("superposition"),
                "latency_ms": result.latency_ms,
                "success": True
            }
        else:
            return {
                "query_id": query_id,
                "response": "",
                "confidence": 0,
                "latency_ms": result.latency_ms,
                "success": False,
                "error": result.error
            }

    def run_baseline_queries(self, queries: List[Dict], results_dir: Path) -> List[Dict]:
        """Run baseline RAG queries"""
        results_dir.mkdir(exist_ok=True)
        results = []

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            from dpr_rc.passive_agent import PassiveWorker
            worker = PassiveWorker()

            for i, query in enumerate(queries):
                query_id = f"query_{i:04d}"

                timestamp_context = query.get("timestamp_context", "")
                if timestamp_context:
                    year = timestamp_context.split("-")[0]
                    shard_id = f"shard_{year}"
                else:
                    shard_id = "shard_2020"

                start = time.perf_counter()
                doc = worker.retrieve(query["question"], shard_id, timestamp_context)
                latency_ms = (time.perf_counter() - start) * 1000

                if doc:
                    results.append({
                        "query_id": query_id,
                        "response": doc["content"],
                        "confidence": 1.0,
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
            results = [{"query_id": f"query_{i:04d}", "success": False} for i in range(len(queries))]

        self._save_json(results, results_dir / "results_baseline.json")
        return results

    def compare_results(
        self,
        queries: List[Dict],
        dprrc_results: List[Dict],
        baseline_results: List[Dict],
        glossary: Dict
    ) -> Dict:
        """Compare DPR-RC vs baseline with superposition-aware evaluation."""
        dprrc_correct = 0
        baseline_correct = 0
        dprrc_hallucinations = []
        baseline_hallucinations = []
        dprrc_latencies = []
        baseline_latencies = []

        hallucination_check_requests = []
        request_metadata = []

        for i, query in enumerate(queries):
            dprrc = dprrc_results[i] if i < len(dprrc_results) else {"success": False}
            baseline = baseline_results[i] if i < len(baseline_results) else {"success": False}

            question_entities = EvaluationService.extract_entities_from_question(
                query["question"]
            )

            ground_truth = {
                "expected_consensus": query.get("expected_consensus", []),
                "expected_disputed": query.get("expected_disputed", [])
            }

            # Evaluate DPR-RC
            if dprrc.get("success"):
                response = dprrc.get("response", "")
                confidence = dprrc.get("confidence", 0)

                eval_config = self._config.get('evaluation', {})
                correctness_result = EvaluationService.evaluate_superposition_correctness(
                    response=response,
                    expected_entities=question_entities,
                    recall_threshold=eval_config.get('recall_threshold', 0.5),
                    min_response_length=eval_config.get('min_response_length', 20),
                    superposition_data=dprrc.get("superposition")
                )

                if correctness_result.is_correct:
                    dprrc_correct += 1

                required_terms = query.get("required_terms", [])
                forbidden_terms = query.get("forbidden_terms", [])

                if required_terms or forbidden_terms:
                    validation_result = EvaluationService.evaluate_with_validation_criteria(
                        response=response,
                        required_terms=required_terms,
                        forbidden_terms=forbidden_terms,
                        validation_pattern=query.get("validation_pattern")
                    )

                    if not validation_result["hallucination_passed"]:
                        dprrc_hallucinations.append({
                            "query_id": dprrc.get("query_id", f"dprrc_{i}"),
                            "type": "forbidden_term",
                            "severity": "high",
                            "explanation": f"Response contains forbidden real-world terms: {', '.join(validation_result['forbidden_found'][:5])}",
                            "flagged_content": validation_result["forbidden_found"][:10],
                            "confidence": confidence,
                            "validation_type": "per_query"
                        })
                else:
                    hallucination_check_requests.append({
                        "query": query.get("question", ""),
                        "system_response": response,
                        "ground_truth": ground_truth,
                        "valid_terms": self.hallucination_detector._extract_valid_terms(glossary),
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

            # Evaluate baseline
            if baseline.get("success"):
                response = baseline.get("response", "")

                eval_config = self._config.get('evaluation', {})
                correctness_result = EvaluationService.evaluate_correctness(
                    response=response,
                    expected_entities=question_entities,
                    recall_threshold=eval_config.get('recall_threshold', 0.5),
                    min_response_length=eval_config.get('min_response_length', 20)
                )

                if correctness_result.is_correct:
                    baseline_correct += 1

                required_terms = query.get("required_terms", [])
                forbidden_terms = query.get("forbidden_terms", [])

                if required_terms or forbidden_terms:
                    validation_result = EvaluationService.evaluate_with_validation_criteria(
                        response=response,
                        required_terms=required_terms,
                        forbidden_terms=forbidden_terms,
                        validation_pattern=query.get("validation_pattern")
                    )

                    if not validation_result["hallucination_passed"]:
                        baseline_hallucinations.append({
                            "query_id": baseline.get("query_id", f"baseline_{i}"),
                            "type": "forbidden_term",
                            "severity": "high",
                            "explanation": f"Response contains forbidden real-world terms: {', '.join(validation_result['forbidden_found'][:5])}",
                            "flagged_content": validation_result["forbidden_found"][:10],
                            "validation_type": "per_query"
                        })
                else:
                    hallucination_check_requests.append({
                        "query": query.get("question", ""),
                        "system_response": response,
                        "ground_truth": ground_truth,
                        "valid_terms": self.hallucination_detector._extract_valid_terms(glossary),
                        "confidence": 1.0,
                        "trace_id": baseline.get("query_id", f"baseline_{i}")
                    })
                    request_metadata.append({
                        "type": "baseline",
                        "index": i,
                        "query_id": baseline.get("query_id")
                    })

                baseline_latencies.append(baseline.get("latency_ms", 0))

        # Batch process hallucination checks
        if hallucination_check_requests:
            batch_results = self.hallucination_detector.batch_detect(hallucination_check_requests)

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
                    else:
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

    def _fetch_audit_trail(self, trace_id: str) -> Dict:
        """Fetch complete audit trail from Cloud Logging"""
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
        """Fetch structured exchange history"""
        try:
            import subprocess
            import sys

            script_path = Path(__file__).parent.parent / "scripts" / "download_query_history.py"
            if not script_path.exists():
                print(f"Warning: Exchange history script not found at {script_path}")
                return {}

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
            if uploaded_count <= 5 or uploaded_count % 10 == 0:
                print(f"  [{uploaded_count}/{len(files_to_upload)}] gs://{bucket_name}/{blob_name}")

        print(f"Uploaded {uploaded_count} files to gs://{bucket_name}/benchmark_results/")
    except Exception as e:
        print(f"ERROR: Could not upload results to GCS: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run complete research benchmark"""
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
        if os.getenv("HISTORY_BUCKET"):
            print("\n--- Uploading results to GCS ---")
            print(f"Bucket: gs://{os.getenv('HISTORY_BUCKET')}/benchmark_results/")
            upload_results_to_gcs(suite.output_dir)
        else:
            print("\nHISTORY_BUCKET not set, skipping GCS upload")


if __name__ == "__main__":
    main()
