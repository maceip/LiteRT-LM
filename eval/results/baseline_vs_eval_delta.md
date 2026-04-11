# Baseline vs Eval Harness — Delta Report

## Unmodified LiteRT-LM Baseline

Raw `litert_lm_main` inference with no eval harness wrapper, no scoring, no memory sampling thread.

| Prompt | Chars | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak RSS (MB) | Mean RSS (MB) | Wall (s) |
|--------|:-----:|:-------------:|:------------:|:---------:|:-------------:|:-------------:|:--------:|
| trivial | 2 | 14.1 | 12.49 | 790 | 2804 | 1460 | 2.2 |
| short | 42 | 28.6 | 13.67 | 700 | 2814 | 2014 | 3.2 |
| medium | 206 | 77.5 | 14.09 | 700 | 3037 | 2928 | 116.9 |
| long | 652 | 46.0 | 12.85 | 5080 | 3447 | 3291 | 95.0 |

## Eval Harness Runs

Inference via Python subprocess with memory sampling thread (0.25s interval), eval scoring, and longer/richer prompts (system prompt + scenario context).

| Harness | Prompt Chars | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak RSS (MB) | Wall (s) | Eval Score |
|---------|:------------:|:-------------:|:------------:|:---------:|:-------------:|:--------:|:----------:|
| Planning (short) | ~600 | 29.4 | 13.00 | 6030 | 3469 | 82.6 | 0.800 |
| Planning (medium) | ~1800 | 71.5 | 14.07 | 5300 | 3489 | 270.7 | 0.675 |
| Planning (long) | ~5500 | 121.9 | 14.09 | 9730 | 3529 | 140.6 | 0.708 |
| Error Diagnosis | ~6800 | 134.4 | 13.87 | 9480 | 3551 | 105.6 | 0.860 |
| Tool Use T1 | ~1200 | 67.0 | 14.27 | 4890 | 3365 | 16.5 | act: 1.0 |
| Tool Use T2 | ~2000 | 182.5 | 14.31 | 4770 | 3401 | 23.0 | act: 0.67 |
| Tool Use T3 | ~3000 | 220.0 | 14.15 | 9290 | 3424 | 22.0 | act: 0.67 |

## Delta Analysis

Comparing closest prompt-length pairs between baseline and eval harness:

| Metric | Baseline (avg) | Eval Harness (avg) | Delta | Notes |
|--------|:--------------:|:------------------:|:-----:|-------|
| **Decode tok/s** | 13.3 | 13.97 | +0.7 (+5%) | No regression; within noise |
| **Prefill tok/s** | 41.6 | 118.1 | +76.5 (+184%) | Longer prompts amortize init overhead |
| **TTFT** | 1818 ms | 7070 ms | +5252 ms | Longer prompts = more prefill work |
| **Peak RSS** | 3025 MB | 3461 MB | +436 MB (+14%) | Longer contexts consume more KV-cache |
| **Init time** | 471 ms | — | — | Constant; not prompt-dependent |

### Key Observations

1. **Decode speed is stable** (~13-14 tok/s) regardless of prompt length or harness overhead. The memory sampling thread adds negligible cost.

2. **Prefill scales with prompt length** as expected: 14 tok/s at 2 chars → 220 tok/s at ~3K chars. The engine batches prefill tokens, so longer prompts see higher throughput.

3. **TTFT increases with context** from 700ms (short) to 9.7s (long). This is dominated by prefill compute, not overhead.

4. **Memory scales with context**: 2.8GB (trivial) → 3.55GB (6.8K chars). The ~700MB delta between shortest and longest is KV-cache allocation.

5. **No eval harness overhead** on core inference metrics. The Python wrapper, memory sampling, and scoring are post-hoc and don't affect the inference process.

## Side-by-Side Summary

```
                    BASELINE              EVAL HARNESS
                    ────────              ────────────
Decode tok/s        12.5 – 14.1           13.0 – 14.3
Prefill tok/s       14.1 – 77.5           29.4 – 220.0
TTFT                700 – 5080 ms         4770 – 9730 ms
Peak RSS            2804 – 3447 MB        3365 – 3551 MB
Model load          422 – 532 ms          (same binary)
```
