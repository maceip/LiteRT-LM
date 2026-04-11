#!/usr/bin/env python3
"""Run all eval harnesses with Gemma 4 E4B and produce a combined report.

Executes:
  1. Agentic Planning evaluation (3 prompt tiers)
  2. Error Diagnosis evaluation (long context)
  3. Multi-Turn Tool Use evaluation (3 turns)

Collects memory profiles, performance metrics, and evaluation scores.
Produces a combined JSON report and a human-readable summary comparing
results to SOTA evaluation benchmarks.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval_agentic_planning import run_planning_eval
from eval_error_diagnosis import run_error_diagnosis_eval
from eval_tool_use import run_tool_use_eval


SOTA_COMPARISON = {
    "description": (
        "Comparison of our on-device Gemma 4 E4B eval against SOTA cloud-based "
        "evaluation frameworks. Note: SOTA benchmarks use cloud APIs with "
        "much larger models (GPT-5.4, Claude Opus 4.6, etc.) while we run "
        "on-device with a 4-bit quantized 4B parameter model."
    ),
    "benchmarks": {
        "SWE-bench_Verified": {
            "description": "500 hand-verified GitHub issues; measure resolve rate",
            "sota_score": "79.2% (GPT-5.4 / Claude Opus 4.6)",
            "metric": "resolve_rate",
            "context_type": "repo codebase + issue description",
            "model_size": "100B+ params (cloud)",
            "our_analog": "eval_error_diagnosis (root cause + fix identification)",
        },
        "SWE-bench_Pro": {
            "description": "1865 harder real-world problems",
            "sota_score": "77.8% (Claude Mythos Preview)",
            "metric": "resolve_rate",
            "context_type": "complex multi-file repos",
            "model_size": "100B+ params (cloud)",
            "our_analog": "eval_tool_use (multi-turn tool-based resolution)",
        },
        "RULER": {
            "description": "NVIDIA long-context synthetic benchmark (NIAH variants)",
            "sota_score": ">95% at 32K (Gemini 1.5 Pro)",
            "metric": "task_accuracy",
            "context_type": "synthetic retrieval / aggregation",
            "model_size": "Various",
            "our_analog": "eval_error_diagnosis (retrieval from build log context)",
        },
        "METR_Time_Horizon": {
            "description": "Autonomous task completion duration",
            "sota_score": "160 min (frontier models)",
            "metric": "completion_time",
            "context_type": "varied autonomous tasks",
            "model_size": "100B+ params (cloud)",
            "our_analog": "eval_agentic_planning (multi-step plan generation)",
        },
        "BinaryAudit": {
            "description": "Security-focused backdoor detection in binaries",
            "sota_score": "49%",
            "metric": "detection_rate",
            "context_type": "binary analysis",
            "model_size": "100B+ (cloud)",
            "our_analog": "N/A (different domain)",
        },
    },
    "key_differences": [
        "Our eval runs entirely on-device (LiteRT-LM) vs cloud API calls",
        "Model size: ~4B params (4-bit quant) vs 100B+ params",
        "Context window: 32K tokens (Gemma 4 E4B) vs 128K-1M (cloud models)",
        "Latency: seconds on CPU vs milliseconds on cloud GPUs",
        "Memory: ~3.5GB model vs unlimited cloud VRAM",
        "Focus: build engineering domain vs general software engineering",
        "Our eval tests practical on-device capability for edge deployment",
    ],
}


def aggregate_memory_profile(all_samples: list) -> dict:
    """Compute aggregate memory statistics."""
    if not all_samples:
        return {"peak_rss_mb": 0, "mean_rss_mb": 0, "min_rss_mb": 0, "samples": 0}

    rss_values = [s[1] for s in all_samples if len(s) >= 2]
    if not rss_values:
        return {"peak_rss_mb": 0, "mean_rss_mb": 0, "min_rss_mb": 0, "samples": 0}

    return {
        "peak_rss_mb": round(max(rss_values), 1),
        "mean_rss_mb": round(sum(rss_values) / len(rss_values), 1),
        "min_rss_mb": round(min(rss_values), 1),
        "samples": len(rss_values),
    }


def aggregate_perf_metrics(results: dict) -> dict:
    """Extract and aggregate performance metrics across all harnesses."""
    metrics = {
        "total_wall_clock_sec": 0,
        "total_inferences": 0,
        "harness_timings": {},
    }

    for harness_name, harness_data in results.items():
        if harness_name in ("metadata", "sota_comparison", "summary"):
            continue

        harness_wall = 0
        harness_inferences = 0

        if isinstance(harness_data, dict):
            if "inference" in harness_data:
                wc = harness_data["inference"].get("metrics", {}).get("wall_clock_sec", 0)
                harness_wall += wc
                harness_inferences += 1
            elif "turns" in harness_data:
                for turn in harness_data["turns"]:
                    wc = turn.get("inference", {}).get("metrics", {}).get("wall_clock_sec", 0)
                    harness_wall += wc
                    harness_inferences += 1
            else:
                for tier_name, tier_data in harness_data.items():
                    if isinstance(tier_data, dict) and "inference" in tier_data:
                        wc = tier_data["inference"].get("metrics", {}).get("wall_clock_sec", 0)
                        harness_wall += wc
                        harness_inferences += 1

        metrics["total_wall_clock_sec"] += harness_wall
        metrics["total_inferences"] += harness_inferences
        metrics["harness_timings"][harness_name] = {
            "wall_clock_sec": round(harness_wall, 3),
            "num_inferences": harness_inferences,
        }

    return metrics


def generate_summary(results: dict) -> str:
    """Generate a human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("GEMMA 4 E4B EVALUATION REPORT - LONG-CONTEXT AGENTIC TASKS")
    lines.append("=" * 70)
    lines.append("")

    planning = results.get("planning", {})
    lines.append("## Harness 1: Agentic Step Planning")
    for tier in ("short", "medium", "long"):
        if tier in planning:
            score = planning[tier].get("score", {})
            lines.append(f"  {tier:>8}: overall={score.get('overall_score', 'N/A'):.3f}  "
                          f"completeness={score.get('step_completeness', 'N/A'):.3f}  "
                          f"ordering={score.get('dependency_ordering', 'N/A'):.3f}  "
                          f"accuracy={score.get('technical_accuracy', 'N/A'):.3f}")
    lines.append("")

    diagnosis = results.get("error_diagnosis", {})
    if "score" in diagnosis:
        ds = diagnosis["score"]
        lines.append("## Harness 2: Error Diagnosis (Long Context)")
        lines.append(f"  overall={ds.get('overall_score', 'N/A'):.3f}  "
                      f"root_cause={ds.get('root_cause_identification', 'N/A'):.3f}  "
                      f"fix={ds.get('fix_quality', 'N/A'):.3f}  "
                      f"evidence={ds.get('evidence_citation', 'N/A'):.3f}")
        lines.append("")

    tool_use = results.get("tool_use", {})
    if "overall_score" in tool_use:
        ts = tool_use["overall_score"]
        lines.append("## Harness 3: Multi-Turn Tool Use")
        lines.append(f"  overall={ts.get('overall_score', 'N/A'):.3f}  "
                      f"actions={ts.get('action_correctness', 'N/A'):.3f}  "
                      f"reasoning={ts.get('reasoning_quality', 'N/A'):.3f}  "
                      f"context={ts.get('context_retention', 'N/A'):.3f}")
        lines.append("")

    mem = results.get("memory_profile", {})
    if mem:
        lines.append("## Memory Profile")
        lines.append(f"  Peak RSS: {mem.get('peak_rss_mb', 0):.1f} MB")
        lines.append(f"  Mean RSS: {mem.get('mean_rss_mb', 0):.1f} MB")
        lines.append(f"  Samples: {mem.get('samples', 0)}")
        lines.append("")

    perf = results.get("performance", {})
    if perf:
        lines.append("## Performance")
        lines.append(f"  Total wall clock: {perf.get('total_wall_clock_sec', 0):.1f}s")
        lines.append(f"  Total inferences: {perf.get('total_inferences', 0)}")
        lines.append("")

    lines.append("## SOTA Comparison")
    lines.append("  See sota_comparison section in JSON output for detailed comparison")
    lines.append("  Key: Our eval runs on-device (4B params, 4-bit quant) vs cloud (100B+)")
    lines.append("")

    return "\n".join(lines)


def run_all(
    model_path: str,
    backend: str = "cpu",
    binary_path: str = None,
    output_dir: str = "eval/results",
):
    """Run all evaluation harnesses."""

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.monotonic()

    results = {
        "metadata": {
            "model": "gemma-4-E4B-it",
            "model_path": model_path,
            "backend": backend,
            "framework": "LiteRT-LM",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_version": "1.0.0",
        }
    }

    all_memory_samples = []

    print("\n" + "=" * 70)
    print("STARTING FULL EVALUATION SUITE")
    print("=" * 70)

    try:
        planning_results = run_planning_eval(model_path, backend, binary_path, output_dir)
        results["planning"] = planning_results
        for tier_data in planning_results.values():
            if isinstance(tier_data, dict) and "memory_samples" in tier_data:
                all_memory_samples.extend(tier_data["memory_samples"])
    except Exception as e:
        print(f"Planning eval failed: {e}")
        results["planning"] = {"error": str(e)}

    try:
        diagnosis_results = run_error_diagnosis_eval(model_path, backend, binary_path, output_dir)
        results["error_diagnosis"] = diagnosis_results
        if "memory_samples" in diagnosis_results:
            all_memory_samples.extend(diagnosis_results["memory_samples"])
    except Exception as e:
        print(f"Error diagnosis eval failed: {e}")
        results["error_diagnosis"] = {"error": str(e)}

    try:
        tool_use_results = run_tool_use_eval(model_path, backend, binary_path, output_dir)
        results["tool_use"] = tool_use_results
        if "turns" in tool_use_results:
            for turn in tool_use_results["turns"]:
                if "memory_samples" in turn:
                    all_memory_samples.extend(turn["memory_samples"])
    except Exception as e:
        print(f"Tool use eval failed: {e}")
        results["tool_use"] = {"error": str(e)}

    results["memory_profile"] = aggregate_memory_profile(all_memory_samples)
    results["performance"] = aggregate_perf_metrics(results)
    results["performance"]["total_wall_clock_sec"] = round(time.monotonic() - start_time, 3)
    results["sota_comparison"] = SOTA_COMPARISON

    summary_text = generate_summary(results)
    results["summary"] = summary_text
    print(summary_text)

    combined_path = Path(output_dir) / "full_eval_report.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report written to {combined_path}")

    summary_path = Path(output_dir) / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Summary written to {summary_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all Gemma 4 E4B eval harnesses")
    parser.add_argument("--model_path", required=True, help="Path to .litertlm model")
    parser.add_argument("--backend", default="cpu", help="Backend (cpu/gpu)")
    parser.add_argument("--binary", default=None, help="Path to litert_lm_main binary")
    parser.add_argument("--output_dir", default="eval/results", help="Output directory")
    args = parser.parse_args()

    run_all(args.model_path, args.backend, args.binary, args.output_dir)
