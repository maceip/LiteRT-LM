#!/usr/bin/env python3
"""Run unmodified LiteRT-LM baseline benchmarks.

Captures raw inference performance and memory for comparison against
the eval harness runs. No scoring -- just perf/memory measurements
at multiple prompt lengths.
"""

import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BaselineMetrics:
    prompt_label: str = ""
    prompt_char_count: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_tok_per_sec: float = 0.0
    decode_tok_per_sec: float = 0.0
    ttft_ms: float = 0.0
    init_time_ms: float = 0.0
    wall_clock_sec: float = 0.0
    peak_rss_mb: float = 0.0
    mean_rss_mb: float = 0.0
    model_output_preview: str = ""


BASELINE_PROMPTS = {
    "trivial": "Hi",
    "short": "What is the tallest building in the world?",
    "medium": (
        "You are a build engineer. Explain step by step how to compile "
        "a large C++ project like Chromium from source on Ubuntu Linux. "
        "Include prerequisite packages, build system setup, and the "
        "actual build commands."
    ),
    "long": (
        "You are an expert systems engineer. I have the following build "
        "failure on Ubuntu 24.04 while compiling Chromium:\n\n"
        "```\n"
        "[31002/52891] LINK ./libcontent.so\n"
        "FAILED: libcontent.so libcontent.so.TOC\n"
        "/usr/bin/ld: final link requires too much memory\n"
        "collect2: error: ld returned 1 exit status\n"
        "[31050/52891] LINK ./libchrome.so\n"
        "FAILED: libchrome.so\n"
        "/usr/bin/ld: cannot find -lstdc++\n"
        "collect2: error: ld returned 1 exit status\n"
        "```\n\n"
        "System has 128GB RAM, GNU ld 2.42, lld-18 is installed but not "
        "configured. GN args: is_debug=true, is_component_build=true, "
        "symbol_level=2, use_lld not set. dmesg shows OOM kill of ld.\n\n"
        "Diagnose the root cause and provide the fix."
    ),
}


def parse_litert_output(text: str) -> dict:
    metrics = {}
    pats = {
        "ttft_sec": r"Time to first token:\s*([\d.]+)\s*s",
        "prefill_tok_s": r"Prefill Speed:\s*([\d.]+)\s*tokens/sec",
        "decode_tok_s": r"Decode Speed:\s*([\d.]+)\s*tokens/sec",
        "init_ms": r"Init Total:\s*([\d.]+)\s*ms",
        "prefill_tokens": r"Prefill Turn \d+:\s*Processed\s+(\d+)\s+tokens",
        "decode_tokens": r"Decode Turn \d+:\s*Processed\s+(\d+)\s+tokens",
    }
    for k, p in pats.items():
        m = re.search(p, text)
        if m:
            metrics[k] = float(m.group(1))
    return metrics


def extract_response(text: str) -> str:
    lines = text.split("\n")
    capture = False
    out = []
    for line in lines:
        if line.startswith("input_prompt:"):
            capture = True
            continue
        if capture:
            if line.startswith("BenchmarkInfo:") or line.startswith("INFO:"):
                break
            out.append(line)
    return "\n".join(out).strip()


def run_one(label: str, prompt: str, model_path: str, binary: str, backend: str) -> BaselineMetrics:
    m = BaselineMetrics(prompt_label=label, prompt_char_count=len(prompt))

    cmd = [binary, f"--backend={backend}", f"--model_path={model_path}", f"--input_prompt={prompt}"]

    memory_samples = []
    wall_start = time.monotonic()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def sampler():
        while proc.poll() is None:
            try:
                parts = Path(f"/proc/{proc.pid}/statm").read_text().split()
                rss_mb = int(parts[1]) * resource.getpagesize() / (1024**2)
                memory_samples.append(rss_mb)
            except Exception:
                pass
            time.sleep(0.25)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    stdout, stderr = proc.communicate(timeout=600)
    wall_sec = time.monotonic() - wall_start
    t.join(timeout=2)

    m.wall_clock_sec = round(wall_sec, 3)

    parsed = parse_litert_output(stdout)
    m.prefill_tok_per_sec = parsed.get("prefill_tok_s", 0)
    m.decode_tok_per_sec = parsed.get("decode_tok_s", 0)
    m.ttft_ms = round(parsed.get("ttft_sec", 0) * 1000, 1)
    m.init_time_ms = parsed.get("init_ms", 0)
    m.prefill_tokens = int(parsed.get("prefill_tokens", 0))
    m.decode_tokens = int(parsed.get("decode_tokens", 0))

    if memory_samples:
        m.peak_rss_mb = round(max(memory_samples), 1)
        m.mean_rss_mb = round(sum(memory_samples) / len(memory_samples), 1)

    m.model_output_preview = extract_response(stdout)[:300]

    return m


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--backend", default="cpu")
    p.add_argument("--binary", default="bazel-bin/runtime/engine/litert_lm_main")
    p.add_argument("--output_dir", default="eval/results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    print(f"{'='*70}")
    print("UNMODIFIED LiteRT-LM BASELINE BENCHMARK")
    print(f"{'='*70}\n")

    for label, prompt in BASELINE_PROMPTS.items():
        print(f"--- {label} ({len(prompt)} chars) ---")
        m = run_one(label, prompt, args.model_path, args.binary, args.backend)
        results.append(asdict(m))
        print(f"  TTFT: {m.ttft_ms}ms | Prefill: {m.prefill_tok_per_sec} tok/s ({m.prefill_tokens} tokens)")
        print(f"  Decode: {m.decode_tok_per_sec} tok/s ({m.decode_tokens} tokens)")
        print(f"  Peak RSS: {m.peak_rss_mb} MB | Mean RSS: {m.mean_rss_mb} MB")
        print(f"  Wall clock: {m.wall_clock_sec}s | Init: {m.init_time_ms}ms")
        print(f"  Output: {m.model_output_preview[:80]}...\n")

    report = {
        "type": "baseline",
        "description": "Unmodified LiteRT-LM inference, no eval scoring overhead",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "gemma-4-E4B-it",
        "backend": args.backend,
        "runs": results,
    }

    out_path = Path(args.output_dir) / "baseline_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved to {out_path}")

    return report


if __name__ == "__main__":
    main()
