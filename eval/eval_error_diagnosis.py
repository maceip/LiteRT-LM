"""Harness 2: Long-Context Error Diagnosis Evaluation.

Tests Gemma 4 E4B's ability to analyze large build error contexts and identify
root causes. This is analogous to how SOTA labs evaluate:
- RULER (NVIDIA): retrieval from long contexts, but we use real build logs
  instead of synthetic needles
- SWE-bench: bug diagnosis, but we focus on build-system errors rather than
  code bugs
- METR: autonomous investigation, but we measure single-shot diagnosis quality

The context window is stressed with ~3-8K tokens of interleaved build output,
config files, system state, and error messages.
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from prompts.chromium_build_context import (
    SYSTEM_PROMPT,
    ERROR_DIAGNOSIS_CONTEXT,
    ERROR_DIAGNOSIS_REFERENCE,
)
from litert_runner import run_inference_with_memory_sampling


@dataclass
class DiagnosisScore:
    root_cause_identification: float = 0.0
    fix_quality: float = 0.0
    evidence_citation: float = 0.0
    overall_score: float = 0.0
    root_causes_found: list = None
    fixes_found: list = None
    evidence_cited: list = None
    false_diagnoses: list = None
    details: str = ""

    def __post_init__(self):
        if self.root_causes_found is None:
            self.root_causes_found = []
        if self.fixes_found is None:
            self.fixes_found = []
        if self.evidence_cited is None:
            self.evidence_cited = []
        if self.false_diagnoses is None:
            self.false_diagnoses = []


ROOT_CAUSE_KEYWORDS = {
    "oom_linker": [
        ("ld", "memory"), ("linker", "OOM"), ("ld", "out of memory"),
        ("GNU ld", "memory"), ("bfd", "memory"),
    ],
    "use_lld_missing": [
        ("use_lld", ""), ("lld", "not enabled"), ("lld", "not using"),
        ("lld", "instead"), ("LLVM", "linker"),
    ],
    "symbol_level_high": [
        ("symbol_level", "2"), ("debug symbols", "large"),
        ("symbol_level", "reduce"), ("symbol_level", "1"),
    ],
    "libstdcpp_missing": [
        ("libstdc++", ""), ("lstdc++", "missing"), ("lstdc++", "not found"),
        ("libstdc++-dev", ""),
    ],
}

FIX_KEYWORDS = {
    "enable_lld": [
        ("use_lld", "true"), ("lld", "enable"), ("-fuse-ld=lld", ""),
    ],
    "reduce_symbols": [
        ("symbol_level", "1"), ("symbol_level", "0"),
        ("blink_symbol_level", "0"),
    ],
    "install_libstdcpp": [
        ("libstdc++", "install"), ("apt", "libstdc++"),
        ("libstdc++-13-dev", ""),
    ],
    "reduce_parallelism": [
        ("-j", ""), ("ninja", "jobs"), ("parallel", "reduce"),
    ],
    "no_keep_memory": [
        ("no-keep-memory", ""), ("keep-memory", ""),
    ],
}

EVIDENCE_KEYWORDS = {
    "dmesg_oom": [
        ("dmesg", "OOM"), ("oom_kill", ""), ("Out of memory", ""),
        ("Killed process", "ld"),
    ],
    "ld_error_msg": [
        ("final link requires too much memory", ""),
        ("recompile with -fno-PIC", ""),
    ],
    "lld_available": [
        ("lld-18", "installed"), ("/usr/bin/lld", ""),
        ("lld", "available"),
    ],
    "memory_state": [
        ("120Gi", "used"), ("free -h", ""), ("1.2Gi", "free"),
    ],
}


def check_keyword_groups(output: str, keyword_map: dict) -> tuple[float, list]:
    """Check which keyword groups are mentioned in the output."""
    output_lower = output.lower()
    found = []

    for group_name, keyword_pairs in keyword_map.items():
        for kw1, kw2 in keyword_pairs:
            kw1_present = kw1.lower() in output_lower
            kw2_present = not kw2 or kw2.lower() in output_lower
            if kw1_present and kw2_present:
                found.append(group_name)
                break

    score = len(found) / len(keyword_map) if keyword_map else 0
    return score, found


def detect_false_diagnoses(output: str) -> list:
    """Detect clearly wrong diagnoses."""
    output_lower = output.lower()
    false_positives = []

    wrong_diagnoses = [
        ("disk space", "The disk has 100GB free, not a disk issue"),
        ("network", "This is a local build, not a network issue"),
        ("permission denied", "No permission errors in the log"),
        ("syntax error", "No syntax errors in the compiler output"),
        ("missing source", "Source is synced, this is a linker issue"),
    ]

    for trigger, reason in wrong_diagnoses:
        if trigger in output_lower:
            ctx_start = max(0, output_lower.find(trigger) - 100)
            ctx_end = min(len(output_lower), output_lower.find(trigger) + 100)
            context = output_lower[ctx_start:ctx_end]
            if "not" not in context and "isn't" not in context:
                false_positives.append(f"False: {trigger} - {reason}")

    return false_positives


def evaluate_diagnosis(output: str) -> DiagnosisScore:
    """Score an error diagnosis response."""
    score = DiagnosisScore()

    rc_score, rc_found = check_keyword_groups(output, ROOT_CAUSE_KEYWORDS)
    score.root_cause_identification = round(rc_score, 3)
    score.root_causes_found = rc_found

    fix_score, fixes_found = check_keyword_groups(output, FIX_KEYWORDS)
    score.fix_quality = round(fix_score, 3)
    score.fixes_found = fixes_found

    ev_score, ev_found = check_keyword_groups(output, EVIDENCE_KEYWORDS)
    score.evidence_citation = round(ev_score, 3)
    score.evidence_cited = ev_found

    score.false_diagnoses = detect_false_diagnoses(output)
    false_penalty = min(0.2, len(score.false_diagnoses) * 0.05)

    weights = {
        "root_cause": 0.40,
        "fix": 0.35,
        "evidence": 0.25,
    }
    raw = (
        score.root_cause_identification * weights["root_cause"]
        + score.fix_quality * weights["fix"]
        + score.evidence_citation * weights["evidence"]
    )
    score.overall_score = round(max(0, raw - false_penalty), 3)

    return score


def run_error_diagnosis_eval(
    model_path: str,
    backend: str = "cpu",
    binary_path: Optional[str] = None,
    output_dir: str = "eval/results",
) -> dict:
    """Run the error diagnosis evaluation."""

    os.makedirs(output_dir, exist_ok=True)

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Analyze the following build failure context and provide:\n"
        "1. Root cause analysis (what is actually causing the failures)\n"
        "2. Specific fixes in priority order\n"
        "3. Evidence from the logs supporting your diagnosis\n\n"
        f"{ERROR_DIAGNOSIS_CONTEXT}"
    )

    print(f"\n{'='*60}")
    print(f"Running error diagnosis eval ({len(full_prompt)} chars)")
    print(f"{'='*60}")

    inference_result, memory_samples = run_inference_with_memory_sampling(
        model_path=model_path,
        prompt=full_prompt,
        backend=backend,
        binary_path=binary_path,
        benchmark=True,
    )

    if inference_result.success:
        diag_score = evaluate_diagnosis(inference_result.output)
    else:
        diag_score = DiagnosisScore(
            details=f"Inference failed: {inference_result.error}"
        )

    result = {
        "score": asdict(diag_score),
        "inference": inference_result.to_dict(),
        "memory_samples": memory_samples,
        "reference": ERROR_DIAGNOSIS_REFERENCE,
    }

    print(f"  Overall score: {diag_score.overall_score}")
    print(f"  Root causes found: {diag_score.root_causes_found}")
    print(f"  Fixes found: {diag_score.fixes_found}")
    print(f"  Evidence cited: {diag_score.evidence_cited}")
    if diag_score.false_diagnoses:
        print(f"  False diagnoses: {diag_score.false_diagnoses}")
    if inference_result.success:
        print(f"  Wall clock: {inference_result.metrics.wall_clock_sec}s")
        print(f"  Peak RSS: {inference_result.metrics.peak_rss_mb}MB")

    out_path = Path(output_dir) / "error_diagnosis_eval.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Error Diagnosis Eval")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--output_dir", default="eval/results")
    args = parser.parse_args()

    run_error_diagnosis_eval(args.model_path, args.backend, args.binary, args.output_dir)
