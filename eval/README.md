# Gemma 4 E4B - Long-Context Agentic Evaluation Harness

## Overview

This evaluation harness tests Gemma 4 E4B (via LiteRT-LM on-device inference) on
**long-context agentic tasks** centered around a realistic software engineering
scenario: **compiling Chromium from source**.

Compiling Chromium is an ideal stress test because it demands:
- Multi-step procedural reasoning across >50 interdependent build steps
- Tool-use planning (depot_tools, gn, autoninja, gclient)
- Error diagnosis with large compiler output (~100K+ token traces)
- Platform-conditional logic (Linux/Mac/Windows, GPU/CPU, component/release)
- Memory management awareness (OOM, swap, ninja -j tuning)

## Evaluation Design

### Harness 1: Agentic Step Planning (`eval_agentic_planning.py`)
Tests the model's ability to produce a correct, ordered plan for compiling
Chromium given a system description. Scored against a reference plan using
step-match accuracy, dependency correctness, and completeness.

### Harness 2: Long-Context Error Diagnosis (`eval_error_diagnosis.py`)
Feeds the model a large context window (~8K-32K tokens) of interleaved build
logs, config files, and error traces. The model must identify the root cause
and propose a fix. Scored on root-cause identification and fix correctness.

### Harness 3: Multi-Turn Tool Use (`eval_tool_use.py`)
Simulates a multi-turn agent loop where the model receives system state,
decides on the next action (run command, edit file, check output), and
continues until the build succeeds or fails. Scored on action correctness
and efficiency (number of turns to resolution).

## Comparison to SOTA Evaluation Methods

| Dimension | Our Harness | SWE-bench | RULER | METR |
|-----------|------------|-----------|-------|------|
| Task type | Agentic build orchestration | GitHub issue resolution | Synthetic retrieval | Autonomous tasks |
| Context length stress | 8K-32K tokens (build logs) | Variable (repo context) | Up to 128K synthetic | Variable |
| Multi-step reasoning | Yes (50+ build steps) | Yes (code changes) | No | Yes |
| Tool use | Simulated shell/file ops | Real code edits | None | Real environment |
| Domain | Systems/build engineering | General software eng | N/A | Varied |
| Metric | Plan accuracy + fix rate | Resolve rate | Task accuracy | Completion rate |
| Edge-device focus | Yes (LiteRT-LM) | No (cloud API) | No (cloud API) | No (cloud API) |

## Running

```bash
# Compile LiteRT-LM first
bazel build //runtime/engine:litert_lm_main

# Download Gemma 4 E4B
python3 eval/download_model.py

# Run all harnesses with memory/perf tracking
python3 eval/run_all.py --model_path models/gemma-4-E4B-it.litertlm --backend cpu

# Run individual harness
python3 eval/eval_agentic_planning.py --model_path models/gemma-4-E4B-it.litertlm
```

## Output

Results are written to `eval/results/` with:
- Per-harness scores (JSON)
- Memory profile (RSS, peak, timeline)
- Performance metrics (prefill tok/s, decode tok/s, TTFT, total latency)
- SOTA comparison table
