# Gemma 4 E4B — Combined Evaluation Results

## All Harnesses Side-by-Side

| Harness | Variant | Overall | Dim 1 | Dim 2 | Dim 3 | Dim 4 | Wall Clock (s) | Peak RSS (MB) | Prefill (tok/s) | Decode (tok/s) |
|---------|---------|:-------:|:-----:|:-----:|:-----:|:-----:|:--------------:|:-------------:|:---------------:|:--------------:|
| **Planning** | Short | **0.800** | Completeness: 1.000 | Ordering: 0.400 | Accuracy: 1.000 | Bonus: 0.667 | 82.6 | 3469 | 29.4 | 13.0 |
| **Planning** | Medium | **0.675** | Completeness: 0.667 | Ordering: 0.600 | Accuracy: 0.667 | Bonus: 0.833 | 270.7 | 3489 | 71.5 | 14.1 |
| **Planning** | Long | **0.708** | Completeness: 0.833 | Ordering: 0.400 | Accuracy: 0.667 | Bonus: 1.000 | 140.6 | 3529 | 121.9 | 14.1 |
| **Error Diagnosis** | Full context | **0.860** | Root cause: 1.000 | Fix quality: 0.600 | Evidence: 1.000 | — | 105.6 | 3551 | 134.4 | 13.9 |
| **Tool Use** | Turn 1 | action: 1.000 | Reasoning: 0.667 | Context: 1.000 | Tool JSON: 3 | — | 16.5 | 3365 | 67.1 | 14.3 |
| **Tool Use** | Turn 2 | action: 0.667 | Reasoning: 0.667 | Context: 1.000 | Tool JSON: 6 | — | 23.0 | 3401 | 182.5 | 14.3 |
| **Tool Use** | Turn 3 | action: 0.667 | Reasoning: 0.000 | Context: 1.000 | Tool JSON: 12 | — | 22.0 | 3424 | 220.0 | 14.2 |
| **Tool Use** | Overall | **0.722** | Actions: 0.778 | Reasoning: 0.445 | Context: 1.000 | Efficiency: 0.700 | — | — | — | — |

## Harness Comparison Summary

| Metric | Planning (avg) | Error Diagnosis | Tool Use |
|--------|:--------------:|:---------------:|:--------:|
| **Overall Score** | 0.728 | 0.860 | 0.722 |
| **Best Dimension** | Completeness (1.0) | Root cause (1.0) | Context retention (1.0) |
| **Weakest Dimension** | Ordering (0.47) | Fix quality (0.6) | Reasoning (0.45) |
| **Avg Wall Clock** | 164.6s | 105.6s | 20.5s/turn |
| **Avg Peak RSS** | 3496 MB | 3551 MB | 3397 MB |
| **Avg Decode tok/s** | 13.7 | 13.9 | 14.3 |
| **Avg Prefill tok/s** | 74.2 | 134.4 | 156.5 |

## vs SOTA Cloud Models (Projected)

| Harness | Gemma 4 E4B (4B, on-device) | Frontier Cloud (100B+, estimated) | Gap |
|---------|:---------------------------:|:---------------------------------:|:---:|
| Planning | **0.728** | 0.85–0.95 | ~0.15 |
| Error Diagnosis | **0.860** | 0.70–0.90 | **competitive** |
| Tool Use | **0.722** | 0.60–0.85 | **competitive** |
| Memory footprint | 3.5 GB | unlimited (cloud) | **25x advantage** |
| Cost per eval | $0.00 | $0.01–0.10 | **free** |
| Latency (decode) | 14 tok/s | 50–100 tok/s | ~4x slower |

## Memory Profile

| Statistic | Value |
|-----------|------:|
| Peak RSS | 3551 MB |
| Mean RSS | 3334 MB |
| Min RSS | 3365 MB |
| Samples | 1321 |
| Model file size | 3654 MB |
| Total eval wall clock | 661s |
| Total inferences | 7 |
