"""Runner for LiteRT-LM inference with Gemma 4 E4B.

Wraps the litert_lm_main binary to run prompts and capture output along with
memory and performance metrics.
"""

import json
import os
import re
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class InferenceMetrics:
    prompt_tokens: int = 0
    output_tokens: int = 0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    peak_rss_mb: float = 0.0
    model_load_time_ms: float = 0.0
    wall_clock_sec: float = 0.0


@dataclass
class InferenceResult:
    prompt: str = ""
    output: str = ""
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)
    success: bool = False
    error: str = ""

    def to_dict(self):
        d = asdict(self)
        return d


def extract_model_response(raw_output: str) -> str:
    """Extract the model's text response from litert_lm_main output.

    The output contains INFO/WARNING log lines, the input prompt echo,
    the model response, and BenchmarkInfo. We extract just the response.
    """
    lines = raw_output.split("\n")
    response_lines = []
    in_response = False
    for line in lines:
        if line.startswith("input_prompt:"):
            in_response = True
            continue
        if in_response:
            if line.startswith("BenchmarkInfo:") or line.startswith("INFO:"):
                break
            response_lines.append(line)
    result = "\n".join(response_lines).strip()
    if not result:
        non_log = [l for l in lines
                   if not l.startswith("INFO:") and not l.startswith("WARNING:")
                   and not l.startswith("DEBUG:") and l.strip()]
        result = "\n".join(non_log).strip()
    return result


def find_litert_binary() -> Optional[Path]:
    """Locate the litert_lm_main binary."""
    candidates = [
        Path("bazel-bin/runtime/engine/litert_lm_main"),
        Path("/workspace/bazel-bin/runtime/engine/litert_lm_main"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


def parse_benchmark_output(text: str) -> dict:
    """Parse benchmark metrics from litert_lm_main output.

    Parses the BenchmarkInfo block produced by the LiteRT-LM engine, e.g.:
      Time to first token: 0.99 s
      Prefill Speed: 20.08 tokens/sec.
      Decode Speed: 11.30 tokens/sec.
      Init Total: 11449.22 ms
    """
    metrics = {}

    patterns = {
        "time_to_first_token_ms": [
            r"Time to first token:\s*([\d.]+)\s*s",
            r"Time to first token:\s*([\d.]+)\s*ms",
        ],
        "prefill_tokens_per_sec": [
            r"Prefill Speed:\s*([\d.]+)\s*tokens/sec",
        ],
        "decode_tokens_per_sec": [
            r"Decode Speed:\s*([\d.]+)\s*tokens/sec",
        ],
        "model_load_time_ms": [
            r"Init Total:\s*([\d.]+)\s*ms",
            r"Init Executor:\s*([\d.]+)\s*ms",
        ],
        "prefill_tokens": [
            r"Processed\s+(\d+)\s+tokens\s+in.*Prefill",
            r"Prefill Turn \d+:\s*Processed\s+(\d+)\s+tokens",
        ],
        "decode_tokens": [
            r"Decode Turn \d+:\s*Processed\s+(\d+)\s+tokens",
        ],
        "prefill_latency_ms": [
            r"Prefill Turn \d+:\s*Processed\s+\d+\s+tokens\s+in\s+([\d.]+)ms",
        ],
        "decode_latency_ms": [
            r"Decode Turn \d+:\s*Processed\s+\d+\s+tokens\s+in\s+([\d.]+)ms",
        ],
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if key == "time_to_first_token_ms" and "s" in pattern and "ms" not in pattern:
                    val *= 1000
                metrics[key] = val
                break

    return metrics


def get_process_peak_rss_mb(pid: int) -> float:
    """Read peak RSS from /proc/{pid}/status."""
    try:
        status_path = Path(f"/proc/{pid}/status")
        if status_path.exists():
            text = status_path.read_text()
            m = re.search(r"VmHWM:\s+(\d+)\s+kB", text)
            if m:
                return int(m.group(1)) / 1024.0
    except (IOError, ValueError):
        pass
    return 0.0


def sample_memory_timeline(pid: int, interval: float = 0.5) -> list:
    """Sample RSS of a process over time. Returns list of (time_sec, rss_mb)."""
    samples = []
    start = time.monotonic()
    try:
        while True:
            rss_path = Path(f"/proc/{pid}/statm")
            if not rss_path.exists():
                break
            parts = rss_path.read_text().split()
            rss_pages = int(parts[1])
            page_size = resource.getpagesize()
            rss_mb = (rss_pages * page_size) / (1024 * 1024)
            elapsed = time.monotonic() - start
            samples.append((round(elapsed, 2), round(rss_mb, 1)))
            time.sleep(interval)
    except (IOError, ValueError, IndexError):
        pass
    return samples


def run_inference(
    model_path: str,
    prompt: str,
    backend: str = "cpu",
    binary_path: Optional[str] = None,
    benchmark: bool = True,
    timeout_sec: int = 600,
) -> InferenceResult:
    """Run a single inference through litert_lm_main and collect metrics."""

    result = InferenceResult(prompt=prompt)

    if binary_path:
        binary = Path(binary_path)
    else:
        binary = find_litert_binary()

    if binary is None or not binary.exists():
        result.error = f"litert_lm_main binary not found (tried {binary})"
        return result

    if not Path(model_path).exists():
        result.error = f"Model file not found: {model_path}"
        return result

    cmd = [
        str(binary),
        f"--backend={backend}",
        f"--model_path={model_path}",
        f"--input_prompt={prompt}",
    ]

    env = os.environ.copy()

    wall_start = time.monotonic()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        stdout, stderr = proc.communicate(timeout=timeout_sec)
        wall_end = time.monotonic()
        wall_sec = wall_end - wall_start

        if proc.returncode != 0:
            result.error = f"Process exited with code {proc.returncode}: {stderr[:2000]}"
            result.output = stdout
            return result

        result.output = extract_model_response(stdout)
        result.success = True
        result.metrics.wall_clock_sec = round(wall_sec, 3)

        parsed = parse_benchmark_output(stdout + stderr)
        for k, v in parsed.items():
            if hasattr(result.metrics, k):
                setattr(result.metrics, k, v)

        if result.metrics.peak_rss_mb == 0:
            peak_key = parsed.get("peak_memory_mb", 0)
            if peak_key:
                result.metrics.peak_rss_mb = peak_key

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        result.error = f"Inference timed out after {timeout_sec}s"
    except Exception as e:
        result.error = f"Inference failed: {e}"

    return result


def run_inference_with_memory_sampling(
    model_path: str,
    prompt: str,
    backend: str = "cpu",
    binary_path: Optional[str] = None,
    benchmark: bool = True,
    timeout_sec: int = 600,
    sample_interval: float = 0.5,
) -> tuple[InferenceResult, list]:
    """Run inference and sample memory in a background thread."""

    import threading

    result_holder = [None]
    memory_samples = []

    binary = Path(binary_path) if binary_path else find_litert_binary()
    if binary is None or not binary.exists():
        r = InferenceResult(prompt=prompt, error="Binary not found")
        return r, []

    cmd = [
        str(binary),
        f"--backend={backend}",
        f"--model_path={model_path}",
        f"--input_prompt={prompt}",
    ]

    wall_start = time.monotonic()

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        def sampler():
            while proc.poll() is None:
                try:
                    rss_path = Path(f"/proc/{proc.pid}/statm")
                    if rss_path.exists():
                        parts = rss_path.read_text().split()
                        rss_pages = int(parts[1])
                        rss_mb = (rss_pages * resource.getpagesize()) / (1024**2)
                        elapsed = time.monotonic() - wall_start
                        memory_samples.append((round(elapsed, 2), round(rss_mb, 1)))
                except (IOError, ValueError, IndexError):
                    pass
                time.sleep(sample_interval)

        t = threading.Thread(target=sampler, daemon=True)
        t.start()

        stdout, stderr = proc.communicate(timeout=timeout_sec)
        t.join(timeout=2)

        wall_sec = time.monotonic() - wall_start

        result = InferenceResult(prompt=prompt, output=extract_model_response(stdout), success=proc.returncode == 0)
        result.metrics.wall_clock_sec = round(wall_sec, 3)

        if proc.returncode != 0:
            result.error = f"Exit code {proc.returncode}: {stderr[:2000]}"

        parsed = parse_benchmark_output(stdout + stderr)
        for k, v in parsed.items():
            if hasattr(result.metrics, k):
                setattr(result.metrics, k, v)

        if memory_samples:
            result.metrics.peak_rss_mb = max(s[1] for s in memory_samples)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        result = InferenceResult(prompt=prompt, error=f"Timed out after {timeout_sec}s")
    except Exception as e:
        result = InferenceResult(prompt=prompt, error=str(e))

    return result, memory_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LiteRT-LM inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", default="What is the tallest building in the world?")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--binary", default=None)
    args = parser.parse_args()

    result = run_inference(args.model_path, args.prompt, args.backend, args.binary)
    print(json.dumps(result.to_dict(), indent=2))
