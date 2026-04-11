"""Harness 3: Multi-Turn Tool Use Evaluation.

Tests Gemma 4 E4B's ability to reason through a multi-turn agent loop for
fixing a Chromium build. Comparable to:
- SWE-bench: multi-step code fixes, but we simulate tool-use interaction
- METR Time Horizon: autonomous task completion with tool access
- YC-Bench: long-horizon planning with tool invocations

Unlike cloud-API evaluations, this runs entirely on-device via LiteRT-LM,
testing the model's ability to maintain context and reason about tool
outputs across multiple turns.
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from prompts.chromium_build_context import SYSTEM_PROMPT, TOOL_USE_SCENARIO
from litert_runner import run_inference_with_memory_sampling


TOOL_USE_SYSTEM = """\
You are a build engineer agent with access to the following tools:
- run_command(cmd): Execute a shell command and return stdout/stderr
- read_file(path): Read the contents of a file
- write_file(path, content): Write content to a file
- search_code(pattern, directory): Search for a pattern in source code

When you want to use a tool, respond with a JSON block:
{"tool": "tool_name", "args": {"arg1": "value1"}}

After receiving tool output, analyze it and decide your next action.
Your goal is to diagnose and fix the build failure.
"""


@dataclass
class ToolUseScore:
    action_correctness: float = 0.0
    reasoning_quality: float = 0.0
    efficiency: float = 0.0
    context_retention: float = 0.0
    overall_score: float = 0.0
    total_turns: int = 0
    correct_actions: int = 0
    total_expected_actions: int = 0
    context_references: int = 0
    details: str = ""


TURN_EXPECTED_KEYWORDS = [
    {
        "actions": [
            ["search", "setAttribute", "grep", "rg"],
            ["read", "args.gn", "cat"],
            ["nm", "symbol", "objdump", "object"],
        ],
        "reasoning": [
            ["linker", "link", "undefined reference"],
            ["component", "shared library", ".so"],
            ["visibility", "export", "symbol"],
        ],
    },
    {
        "actions": [
            ["gn", "desc", "deps"],
            ["BUILD.gn", "build file"],
            ["COMPONENT_EXPORT", "export", "visibility", "macro"],
        ],
        "reasoning": [
            ["deps", "dependency", "depend"],
            ["visibility", "export", "hidden"],
            ["BUILD.gn", "build graph"],
        ],
    },
    {
        "actions": [
            ["free", "memory", "top", "htop", "vmstat"],
            ["-j", "jobs", "parallel"],
            ["swap", "swapon", "mkswap"],
        ],
        "reasoning": [
            ["OOM", "out of memory", "memory pressure"],
            ["swap", "swapfile"],
            ["reduce", "limit", "throttle", "fewer"],
        ],
    },
]


def score_turn_response(output: str, turn_idx: int) -> dict:
    """Score a single turn's response."""
    output_lower = output.lower()
    expected = TURN_EXPECTED_KEYWORDS[turn_idx] if turn_idx < len(TURN_EXPECTED_KEYWORDS) else None

    if not expected:
        return {"action_score": 0, "reasoning_score": 0, "actions_found": [], "reasoning_found": []}

    actions_found = []
    for action_group in expected["actions"]:
        if any(kw.lower() in output_lower for kw in action_group):
            actions_found.append(action_group[0])

    reasoning_found = []
    for reason_group in expected["reasoning"]:
        if any(kw.lower() in output_lower for kw in reason_group):
            reasoning_found.append(reason_group[0])

    action_score = len(actions_found) / len(expected["actions"]) if expected["actions"] else 0
    reasoning_score = len(reasoning_found) / len(expected["reasoning"]) if expected["reasoning"] else 0

    return {
        "action_score": round(action_score, 3),
        "reasoning_score": round(reasoning_score, 3),
        "actions_found": actions_found,
        "reasoning_found": reasoning_found,
    }


def check_tool_json_usage(output: str) -> int:
    """Check if model attempts to use tool-call JSON format."""
    import re
    tool_patterns = [
        r'\{"tool":\s*"[^"]+",\s*"args"',
        r'"tool":\s*"(run_command|read_file|write_file|search_code)"',
        r'```json\s*\{[^}]*"tool"',
    ]
    count = 0
    for pattern in tool_patterns:
        count += len(re.findall(pattern, output, re.IGNORECASE))
    return count


def check_context_retention(output: str, turn_idx: int) -> float:
    """Check if the model references information from prior context."""
    output_lower = output.lower()
    context_refs = 0

    if turn_idx == 0:
        if any(kw in output_lower for kw in ["undefined reference", "setattribute", "blink"]):
            context_refs += 1
        if any(kw in output_lower for kw in ["component", "out/default"]):
            context_refs += 1
        return min(1.0, context_refs / 2)

    if turn_idx == 1:
        if any(kw in output_lower for kw in ["setattribute", "element.o"]):
            context_refs += 1
        if any(kw in output_lower for kw in ["libcontent", "content"]):
            context_refs += 1
        if any(kw in output_lower for kw in ["symbol", "nm"]):
            context_refs += 1
        return min(1.0, context_refs / 3)

    if turn_idx == 2:
        if any(kw in output_lower for kw in ["build.gn", "dep", "dependency"]):
            context_refs += 1
        if any(kw in output_lower for kw in ["42000", "52891", "progress"]):
            context_refs += 1
        if any(kw in output_lower for kw in ["95%", "memory", "ram"]):
            context_refs += 1
        return min(1.0, context_refs / 3)

    return 0.0


def run_tool_use_eval(
    model_path: str,
    backend: str = "cpu",
    binary_path: Optional[str] = None,
    output_dir: str = "eval/results",
) -> dict:
    """Run multi-turn tool-use evaluation.

    Since LiteRT-LM runs single-shot inference (no stateful conversation API
    from CLI), we simulate multi-turn by concatenating conversation history
    into increasingly longer prompts.
    """
    os.makedirs(output_dir, exist_ok=True)

    scenario = TOOL_USE_SCENARIO
    turns_data = scenario["turns"]
    conversation_history = f"{TOOL_USE_SYSTEM}\n\nScenario: {scenario['description']}\n"
    conversation_history += f"Initial state: {json.dumps(scenario['initial_state'], indent=2)}\n\n"

    all_turn_results = []
    all_memory_samples = []
    total_action_score = 0
    total_reasoning_score = 0
    total_context_score = 0

    for i, turn in enumerate(turns_data):
        conversation_history += f"[Turn {turn['turn']}] User: {turn['user_message']}\n\n"
        conversation_history += "Assistant: "

        print(f"\n{'='*60}")
        print(f"Running tool-use eval turn {turn['turn']} ({len(conversation_history)} chars)")
        print(f"{'='*60}")

        inference_result, mem_samples = run_inference_with_memory_sampling(
            model_path=model_path,
            prompt=conversation_history,
            backend=backend,
            binary_path=binary_path,
            benchmark=True,
        )

        if inference_result.success:
            turn_score = score_turn_response(inference_result.output, i)
            ctx_score = check_context_retention(inference_result.output, i)
            tool_json_count = check_tool_json_usage(inference_result.output)

            total_action_score += turn_score["action_score"]
            total_reasoning_score += turn_score["reasoning_score"]
            total_context_score += ctx_score

            conversation_history += inference_result.output + "\n\n"
        else:
            turn_score = {"action_score": 0, "reasoning_score": 0,
                          "actions_found": [], "reasoning_found": []}
            ctx_score = 0
            tool_json_count = 0
            conversation_history += "[inference failed]\n\n"

        turn_result = {
            "turn": turn["turn"],
            "scores": turn_score,
            "context_retention": round(ctx_score, 3),
            "tool_json_attempts": tool_json_count,
            "inference": inference_result.to_dict(),
            "memory_samples": mem_samples,
        }
        all_turn_results.append(turn_result)
        all_memory_samples.extend(mem_samples)

        print(f"  Action score: {turn_score['action_score']}")
        print(f"  Reasoning score: {turn_score['reasoning_score']}")
        print(f"  Context retention: {ctx_score}")
        print(f"  Tool JSON attempts: {tool_json_count}")
        if inference_result.success:
            print(f"  Wall clock: {inference_result.metrics.wall_clock_sec}s")
            print(f"  Peak RSS: {inference_result.metrics.peak_rss_mb}MB")

    n_turns = len(turns_data)
    overall = ToolUseScore(
        action_correctness=round(total_action_score / n_turns, 3) if n_turns else 0,
        reasoning_quality=round(total_reasoning_score / n_turns, 3) if n_turns else 0,
        efficiency=round(1.0 - (n_turns / scenario["max_turns"]), 3),
        context_retention=round(total_context_score / n_turns, 3) if n_turns else 0,
        total_turns=n_turns,
    )

    weights = {"action": 0.30, "reasoning": 0.30, "context": 0.25, "efficiency": 0.15}
    overall.overall_score = round(
        overall.action_correctness * weights["action"]
        + overall.reasoning_quality * weights["reasoning"]
        + overall.context_retention * weights["context"]
        + overall.efficiency * weights["efficiency"],
        3,
    )

    result = {
        "overall_score": asdict(overall),
        "turns": all_turn_results,
        "scenario": {
            "description": scenario["description"],
            "max_turns": scenario["max_turns"],
        },
    }

    out_path = Path(output_dir) / "tool_use_eval.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tool Use Eval")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--output_dir", default="eval/results")
    args = parser.parse_args()

    run_tool_use_eval(args.model_path, args.backend, args.binary, args.output_dir)
