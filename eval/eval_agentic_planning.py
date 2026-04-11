"""Harness 1: Agentic Step Planning Evaluation.

Tests Gemma 4 E4B's ability to produce a correct, ordered plan for compiling
Chromium. Evaluates plan quality using:
- Step completeness (coverage of required steps)
- Dependency ordering correctness
- Technical accuracy of commands
- Practical considerations (disk, memory, timing)

Inspired by SWE-bench's task decomposition scoring and METR's multi-step
agentic evaluations, but focused on build-system orchestration rather than
code patching.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from prompts.chromium_build_context import (
    SYSTEM_PROMPT,
    PLANNING_PROMPT_SHORT,
    PLANNING_PROMPT_MEDIUM,
    PLANNING_PROMPT_LONG,
    PLANNING_REFERENCE_STEPS,
)
from litert_runner import run_inference_with_memory_sampling, InferenceResult


REQUIRED_CONCEPTS = {
    "prerequisites": [
        "apt", "install", "build-essential", "python3", "git", "curl",
        "lsb-release",
    ],
    "depot_tools": [
        "depot_tools", "clone", "PATH",
    ],
    "gclient": [
        "gclient", "sync", "fetch", "chromium",
    ],
    "gn_gen": [
        "gn", "gen", "args.gn", "out/",
    ],
    "build": [
        "autoninja", "ninja", "chrome",
    ],
    "verification": [
        "chrome", "--version", "test", "verify", "run",
    ],
}

DEPENDENCY_ORDER = [
    "prerequisites",
    "depot_tools",
    "gclient",
    "gn_gen",
    "build",
    "verification",
]

BONUS_CONCEPTS = {
    "memory_management": ["swap", "OOM", "memory", "-j", "ulimit"],
    "ccache": ["ccache", "cache"],
    "disk_management": ["df", "disk", "space", "clean", "free"],
    "timing": ["time", "elapsed", "duration", "hours", "minutes"],
    "cross_compile": ["arm64", "aarch64", "cross", "target_cpu"],
    "error_handling": ["error", "fail", "rollback", "retry", "check"],
}


@dataclass
class PlanningScore:
    prompt_tier: str = ""
    step_completeness: float = 0.0
    dependency_ordering: float = 0.0
    technical_accuracy: float = 0.0
    bonus_coverage: float = 0.0
    overall_score: float = 0.0
    steps_found: list = None
    steps_missing: list = None
    ordering_violations: list = None
    bonus_found: list = None
    details: str = ""

    def __post_init__(self):
        if self.steps_found is None:
            self.steps_found = []
        if self.steps_missing is None:
            self.steps_missing = []
        if self.ordering_violations is None:
            self.ordering_violations = []
        if self.bonus_found is None:
            self.bonus_found = []


def score_step_completeness(output: str) -> tuple[float, list, list]:
    """Check which required concept groups appear in the output."""
    output_lower = output.lower()
    found = []
    missing = []

    for group_name, keywords in REQUIRED_CONCEPTS.items():
        group_found = sum(1 for kw in keywords if kw.lower() in output_lower)
        if group_found >= 2:
            found.append(group_name)
        else:
            missing.append(group_name)

    score = len(found) / len(REQUIRED_CONCEPTS) if REQUIRED_CONCEPTS else 0
    return score, found, missing


def score_dependency_ordering(output: str) -> tuple[float, list]:
    """Check that steps appear in the correct dependency order."""
    output_lower = output.lower()
    violations = []

    positions = {}
    for group_name, keywords in REQUIRED_CONCEPTS.items():
        first_pos = len(output_lower)
        for kw in keywords:
            pos = output_lower.find(kw.lower())
            if pos >= 0:
                first_pos = min(first_pos, pos)
        if first_pos < len(output_lower):
            positions[group_name] = first_pos

    for i in range(len(DEPENDENCY_ORDER) - 1):
        a, b = DEPENDENCY_ORDER[i], DEPENDENCY_ORDER[i + 1]
        if a in positions and b in positions:
            if positions[a] > positions[b]:
                violations.append(f"{a} appears after {b}")

    if not positions:
        return 0.0, ["no steps detected"]

    n_pairs = sum(
        1 for i in range(len(DEPENDENCY_ORDER) - 1)
        if DEPENDENCY_ORDER[i] in positions and DEPENDENCY_ORDER[i + 1] in positions
    )
    if n_pairs == 0:
        return 0.5, []

    score = 1.0 - (len(violations) / n_pairs)
    return max(0, score), violations


def score_technical_accuracy(output: str) -> float:
    """Heuristic check for technically correct commands."""
    output_lower = output.lower()
    checks = [
        "git clone" in output_lower or "fetch" in output_lower,
        "depot_tools" in output_lower,
        ("gn gen" in output_lower or "gn args" in output_lower),
        ("autoninja" in output_lower or "ninja -C" in output_lower),
        any(x in output_lower for x in ["args.gn", "is_component_build", "is_debug"]),
        any(x in output_lower for x in ["--backend", "out/", "target_cpu"]),
    ]
    return sum(checks) / len(checks)


def score_bonus_concepts(output: str) -> tuple[float, list]:
    """Check for bonus practical considerations."""
    output_lower = output.lower()
    found = []
    for concept_name, keywords in BONUS_CONCEPTS.items():
        if any(kw.lower() in output_lower for kw in keywords):
            found.append(concept_name)
    score = len(found) / len(BONUS_CONCEPTS) if BONUS_CONCEPTS else 0
    return score, found


def evaluate_planning_output(output: str, prompt_tier: str) -> PlanningScore:
    """Score a planning response."""
    score = PlanningScore(prompt_tier=prompt_tier)

    comp, found, missing = score_step_completeness(output)
    score.step_completeness = round(comp, 3)
    score.steps_found = found
    score.steps_missing = missing

    order, violations = score_dependency_ordering(output)
    score.dependency_ordering = round(order, 3)
    score.ordering_violations = violations

    score.technical_accuracy = round(score_technical_accuracy(output), 3)

    bonus, bonus_found = score_bonus_concepts(output)
    score.bonus_coverage = round(bonus, 3)
    score.bonus_found = bonus_found

    weights = {
        "step_completeness": 0.35,
        "dependency_ordering": 0.25,
        "technical_accuracy": 0.25,
        "bonus_coverage": 0.15,
    }
    score.overall_score = round(
        score.step_completeness * weights["step_completeness"]
        + score.dependency_ordering * weights["dependency_ordering"]
        + score.technical_accuracy * weights["technical_accuracy"]
        + score.bonus_coverage * weights["bonus_coverage"],
        3,
    )

    return score


def run_planning_eval(
    model_path: str,
    backend: str = "cpu",
    binary_path: Optional[str] = None,
    output_dir: str = "eval/results",
) -> dict:
    """Run the full planning evaluation across all prompt tiers."""

    os.makedirs(output_dir, exist_ok=True)

    prompts = {
        "short": PLANNING_PROMPT_SHORT,
        "medium": PLANNING_PROMPT_MEDIUM,
        "long": PLANNING_PROMPT_LONG,
    }

    results = {}

    for tier, prompt in prompts.items():
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        print(f"\n{'='*60}")
        print(f"Running planning eval: {tier} ({len(full_prompt)} chars)")
        print(f"{'='*60}")

        inference_result, memory_samples = run_inference_with_memory_sampling(
            model_path=model_path,
            prompt=full_prompt,
            backend=backend,
            binary_path=binary_path,
            benchmark=True,
        )

        if inference_result.success:
            plan_score = evaluate_planning_output(inference_result.output, tier)
        else:
            plan_score = PlanningScore(
                prompt_tier=tier,
                details=f"Inference failed: {inference_result.error}",
            )

        results[tier] = {
            "score": asdict(plan_score),
            "inference": inference_result.to_dict(),
            "memory_samples": memory_samples,
        }

        print(f"  Overall score: {plan_score.overall_score}")
        print(f"  Steps found: {plan_score.steps_found}")
        print(f"  Steps missing: {plan_score.steps_missing}")
        if inference_result.success:
            print(f"  Wall clock: {inference_result.metrics.wall_clock_sec}s")
            print(f"  Peak RSS: {inference_result.metrics.peak_rss_mb}MB")

    out_path = Path(output_dir) / "planning_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agentic Planning Eval")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--output_dir", default="eval/results")
    args = parser.parse_args()

    run_planning_eval(args.model_path, args.backend, args.binary, args.output_dir)
