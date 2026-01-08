"""
Report Generator Module

Generates publication-ready markdown reports for benchmark results.
"""

import time
from pathlib import Path
from typing import Dict


class ReportGenerator:
    """Generates research reports from benchmark results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate(self, results_summary: Dict):
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

            self._write_results_table(f, results_summary)
            self._write_latency_comparison(f, results_summary)
            self._write_key_findings(f, results_summary)
            self._write_reproducibility(f)
            self._write_audit_structure(f)

        print(f"\n✓ Research report generated: {report_path}")

    def _write_results_table(self, f, results_summary: Dict):
        """Write results by scale table"""
        f.write("## Results by Scale\n\n")
        f.write("| Scale | Events | Queries | DPR-RC Accuracy | Baseline Accuracy | DPR-RC Halluc. | Baseline Halluc. |\n")
        f.write("|-------|--------|---------|-----------------|-------------------|----------------|------------------|\n")

        for result in results_summary.get("scale_results", []):
            f.write(f"| {result['scale']} | ")
            f.write(f"{result['dataset_size']} | ")
            f.write(f"{result['query_count']} | ")
            f.write(f"{result['dprrc_accuracy']:.2%} | ")
            f.write(f"{result['baseline_accuracy']:.2%} | ")
            f.write(f"{result['dprrc_hallucination_rate']:.2%} | ")
            f.write(f"{result['baseline_hallucination_rate']:.2%} |\n")

    def _write_latency_comparison(self, f, results_summary: Dict):
        """Write latency comparison table"""
        f.write("\n## Latency Comparison\n\n")
        f.write("| Scale | DPR-RC Mean (ms) | Baseline Mean (ms) | Overhead |\n")
        f.write("|-------|------------------|--------------------|-----------|\n")

        for result in results_summary.get("scale_results", []):
            dprrc_lat = result['dprrc_mean_latency_ms']
            baseline_lat = result['baseline_mean_latency_ms']
            overhead = ((dprrc_lat / baseline_lat) - 1) * 100 if baseline_lat > 0 else 0

            f.write(f"| {result['scale']} | ")
            f.write(f"{dprrc_lat:.1f} | ")
            f.write(f"{baseline_lat:.1f} | ")
            f.write(f"+{overhead:.1f}% |\n")

    def _write_key_findings(self, f, results_summary: Dict):
        """Write key findings section"""
        f.write("\n## Key Findings\n\n")

        for i, result in enumerate(results_summary.get("scale_results", [])):
            if result['baseline_hallucination_rate'] > 0.1 and result['dprrc_hallucination_rate'] < 0.05:
                f.write(f"**Hallucination Threshold**: At {result['scale']} scale ")
                f.write(f"({result['dataset_size']} events), baseline hallucination rate ")
                f.write(f"reaches {result['baseline_hallucination_rate']:.1%} while DPR-RC ")
                f.write(f"maintains {result['dprrc_hallucination_rate']:.1%}.\n\n")
                break

    def _write_reproducibility(self, f):
        """Write reproducibility section"""
        f.write("## Reproducibility\n\n")
        f.write("All artifacts for peer review are available in:\n")
        f.write(f"- `{self.output_dir}/`\n")
        f.write("  - `{scale}/dataset.json` - Generated datasets\n")
        f.write("  - `{scale}/glossary.json` - Phonotactic term definitions\n")
        f.write("  - `{scale}/dprrc_results/query_XXXX/` - Per-query audit trails\n")
        f.write("  - `{scale}/comparison.json` - Detailed evaluation metrics\n\n")

    def _write_audit_structure(self, f):
        """Write audit trail structure section"""
        f.write("## Audit Trail Structure\n\n")
        f.write("Each DPR-RC query includes:\n")
        f.write("- `input.json` - Query text and context\n")
        f.write("- `ground_truth.json` - Expected consensus/disputed claims\n")
        f.write("- `system_output.json` - Final response and confidence\n")
        f.write("- `audit_trail.json` - Complete L1/L2/L3 execution trace\n\n")
