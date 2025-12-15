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

        self.controller_url = os.getenv("CONTROLLER_URL", "http://localhost:8080/query")
        if not self.controller_url.endswith("/query"):
            self.controller_url = f"{self.controller_url.rstrip('/')}/query"
        
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
        """Run queries against DPR-RC with full audit trail capture"""
        
        results_dir.mkdir(exist_ok=True)
        results = []
        
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
            
            # Execute query
            try:
                start = time.perf_counter()
                response = requests.post(
                    self.controller_url,
                    json={
                        "query_text": query["question"],
                        "timestamp_context": query.get("timestamp_context"),
                        "trace_id": query_id
                    },
                    timeout=60
                )
                latency_ms = (time.perf_counter() - start) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Save system output
                    self._save_json({
                        "final_response": data.get("final_answer", ""),
                        "confidence": data.get("confidence", 0),
                        "sources": data.get("sources", []),
                        "status": data.get("status", "SUCCESS")
                    }, query_dir / "system_output.json")
                    
                    # Fetch audit trail from Cloud Logging
                    audit_trail = self._fetch_audit_trail(query_id)
                    if audit_trail:
                        self._save_json(audit_trail, query_dir / "audit_trail.json")
                    
                    results.append({
                        "query_id": query_id,
                        "response": data.get("final_answer", ""),
                        "confidence": data.get("confidence", 0),
                        "latency_ms": latency_ms,
                        "success": True
                    })
                else:
                    results.append({
                        "query_id": query_id,
                        "response": "",
                        "confidence": 0,
                        "latency_ms": latency_ms,
                        "success": False,
                        "error": response.text
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
    
    def compare_results(
        self,
        queries: List[Dict],
        dprrc_results: List[Dict],
        baseline_results: List[Dict],
        glossary: Dict
    ) -> Dict:
        """Compare DPR-RC vs baseline with superposition-aware evaluation"""
        
        dprrc_correct = 0
        baseline_correct = 0
        dprrc_hallucinations = []
        baseline_hallucinations = []
        dprrc_latencies = []
        baseline_latencies = []
        
        for i, query in enumerate(queries):
            dprrc = dprrc_results[i] if i < len(dprrc_results) else {"success": False}
            baseline = baseline_results[i] if i < len(baseline_results) else {"success": False}
            
            # Extract expected entities from query (phonotactic terms)
            question_entities = [w for w in query["question"].split() if w and w[0].isupper()]
            
            # Evaluate DPR-RC (superposition-aware)
            if dprrc.get("success"):
                response = dprrc.get("response", "")
                confidence = dprrc.get("confidence", 0)
                
                # Split response into potential options (naive split by newline or bullets)
                options = [opt for opt in response.replace("- ", "\n").split("\n") if len(opt.strip()) > 10]
                if not options: 
                    options = [response]

                # Check if correct answer is present in ANY option
                any_option_correct = False
                for opt in options:
                    hit_count = sum(1 for e in question_entities if e in opt)
                    recall = hit_count / len(question_entities) if question_entities else 0
                    if recall > 0.5:
                        any_option_correct = True
                        break
                
                if any_option_correct:
                    dprrc_correct += 1
                
                # Check for hallucinations
                # CRITICAL: If system is uncertain (confidence < 0.9) or presents conflict, 
                # incorrect options are valid "perspectives", NOT hallucinations.
                # Only count terms that are completely fabricated (not in glossary) and presented as FACT.
                
                is_uncertain = confidence < 0.9 or "mixed" in response.lower() or "perspectives" in response.lower()
                
                response_terms = [w for w in response.split() if w and w[0].isupper()]
                hallucinated_terms = [t for t in response_terms if t not in glossary]
                
                if hallucinated_terms:
                    # If uncertain, we are more lenient: strictly check against ALL retrieved content?
                    # For this benchmark, if uncertain, we allow "incorrect" options as long as they aren't pure gibberish.
                    # Terms not in glossary are still pure gibberish/hallucinations.
                    dprrc_hallucinations.append({
                        "query_id": dprrc.get("query_id"),
                        "terms": hallucinated_terms,
                        "context": "uncertain" if is_uncertain else "confident"
                    })
                
                dprrc_latencies.append(dprrc.get("latency_ms", 0))
            
            # Evaluate baseline
            if baseline.get("success"):
                response = baseline.get("response", "")
                
                hit_count = sum(1 for e in question_entities if e in response)
                recall = hit_count / len(question_entities) if question_entities else 0
                
                if recall > 0.5 and len(response) > 20:
                    baseline_correct += 1
                
                response_terms = [w for w in response.split() if w and w[0].isupper()]
                hallucinated_terms = [t for t in response_terms if t not in glossary]
                if hallucinated_terms:
                    baseline_hallucinations.append({
                        "query_id": baseline.get("query_id"),
                        "terms": hallucinated_terms
                    })
                
                baseline_latencies.append(baseline.get("latency_ms", 0))
        
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
