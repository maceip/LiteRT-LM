# Evolution

How LiteRT-LM's runtime memory policy works, why it's built this way,
and how frontier models inform the design.

---

## Three Layers

The runtime memory policy is three layers stacked on top of each other.
Each layer depends on the one below it. On failure, each layer falls back
to the one below.

```
  Native Cache ── engine-native KV surgery (Pin, Evict, Remap, etc.)
  │  Capability-gated. Falls back to Prefetch on any failure.
  ├─────────────────────────────────────────────────────────────────
  Prefetch ── background planning + precomputed replay packs
  │  Falls back to synchronous replay on mismatch.
  ├─────────────────────────────────────────────────────────────────
  Safety ── atomic turns, boundary queueing, priority arbitration
  │  Always enforced. Never bypassed.
  └─────────────────────────────────────────────────────────────────
```

| Layer | What it does | Gate doc |
|:------|:-------------|:---------|
| **Safety** | Policy changes only apply at turn/tool boundaries, never mid-inference. Profile constraints override runtime overrides. | `PHASE_A_GATE.md` |
| **Prefetch** | When context nears the shift threshold, a background planner builds a replay pack. If it's still fresh at the boundary, it installs instantly instead of full recompute. | `PHASE_B_GATE.md` |
| **Native Cache** | Engine-native block-level KV ops (Pin, EvictRange, Remap, Compact, SnapshotRestore). Only used when the engine explicitly advertises capability. Atomic at the op-group level. | `PHASE_C_BOOTSTRAP.md`, `PHASE_C_CACHE_OPS_RFC_DRAFT.md` |

---

## Why This Architecture

Three constraints drove the shift from a simple session/checkpoint model:

**Privacy.** On-device inference keeps data on the user's hardware. The layered
architecture makes this auditable: the Safety layer guarantees policy
transitions are atomic and observable; the Prefetch layer tracks exactly which
conversation segments are retained or discarded; the Native Cache layer makes
retention semantics explicit per cache block via pin classes (`system_anchor`,
`attention_sink`, `protected_tail`, `tool_state`, `ephemeral`).

**User control.** The middleware — not the engine — chooses the memory strategy.
Profile constraints (`allow_runtime_tuning`, `safe_boundary`) let applications
enforce policies the runtime must respect. The engine executes; it never
unilaterally decides what to keep or discard.

**Performance.** Context shifts go from full-recompute spikes to background
planning (Prefetch) or sub-linear native ops (Native Cache). But performance
gains are only safe because the Safety layer prevents mid-turn corruption.

---

## How Frontier Models Manage KV Cache

This comparison grounds the architecture in what current models actually do.
See `PHASE_C_CACHE_OPS_RFC_DRAFT.md` for the full evidence basis.

| Model | Params (total/active) | Context | KV strategy | Relevance to LiteRT-LM |
|:------|:----------------------|:--------|:------------|:-----------------------|
| **Gemma 4** 26B-A4B | 26B / ~4B | 128-256K | Hybrid SWA/Full + GQA | Primary target. Predictable SWA triggers benefit Prefetch planning. |
| **Gemma 4** 31B | 31B / 31B | 256K | Hybrid SWA/Full, 16 KV / 32 Q heads | Dense variant; same attention pattern. |
| **Step 3.5 Flash** | 196B / ~11B | 256K | Hybrid SWA/Full (3:1), w=512 | Similar two-tier cache behavior. |
| **Qwen3-235B** | 235B / ~22B | 262K-1M | GQA (4 KV / 64 Q heads, 93.75% reduction) | GQA blocks map directly to Native Cache ops. |
| **DeepSeek V3.2** | 685B / ~37B | 128K | MLA (~98% latent compression) | Low cache pressure; Prefetch triggers rarely but fallback still needed. |
| **Llama 4 Scout** | 109B / ~17B | 10M | iRoPE (interleaved ±RoPE) | Linear KV growth; Native Cache eviction/compaction most valuable here. |
| **Llama 4 Maverick** | 400B / ~17B | 1M | iRoPE | Same as Scout, smaller context. |
| **GLM-4.5** | 355B / ~32B | 128K | API-level context caching | Architecture undisclosed. |
| **Muse Spark** | undisclosed | n/a | Thought compression (2.7x fewer tokens than Claude Opus 4.6) | Indirect cache relief. |
| **GPT-4.1** | undisclosed | 1M | API-level prompt caching (prefix reuse) | Architecture undisclosed. |
| **Grok 4.1** | ~3T / undisclosed | 256K | Undisclosed | Architecture undisclosed. |

Four cache-pressure strategies emerge:

1. **Architectural compression** (build-time): GQA head sharing, MLA latent projection, hybrid SWA window eviction
2. **Positional innovation** (train-time): iRoPE interleaved layers, decoupled RoPE, partial RoPE
3. **Inference-time optimization** (serve-time): prefix caching, PagedAttention, ring-buffer allocation
4. **Reasoning compression** (model-time): thought compression to reduce token count

Patterns: 8/10 open-weight models are MoE. Hybrid attention (SWA + full) is
the norm. Every model emphasizes tool calling or multi-agent orchestration —
long-lived agentic sessions with many turn boundaries, which is exactly what
the Safety layer's boundary-safe transitions are designed for.

---

## System Architecture

```
  Application (Kotlin / Python / C++ / C)
       │
       ▼
  Conversation Runtime
  ├── Memory Policy Subsystem
  │   ├── Safety Layer (boundary queue, atomic turn, priority arbiter)
  │   ├── Prefetch Layer (planner, replay packs, telemetry)
  │   └── Native Cache Layer (capability discovery, CacheOpGroup, rollback)
  │
  ├── Policy Arbiter (profile > runtime > limits)
  └── Context Shift Engine (strategy exec, retained slices, replay, checkpoints)
       │
       ▼
  Engine / Executor
  ├── RunPrefill, RunDecode, Clone, Checkpoint Save/Rewind
  └── KV Cache (block_id, token_span, lineage, pin_class, heat_score)
       │
       ▼
  Hardware (GPU / NPU / CPU)
```

Context shift data flow:

1. Context usage exceeds threshold
2. Safety layer: turn active? → queue and wait for boundary
3. Safety layer: priority arbitration (profile > runtime > limits)
4. Prefetch layer: valid pack? → install at boundary (fast) or sync replay (baseline)
5. Native Cache layer: engine capability? → CacheOpGroup (fastest) or fall back to Prefetch
6. Emit telemetry (`profile_id`, `strategy`, `builder_id`, `boundary`, `model_type`, `reason_code`)
7. Resume inference

---

## Gemma 4 Evaluation

Gemma 4 is the primary evaluation target because it runs in Google products
(Chrome, Chromebook Plus, Pixel Watch) via LiteRT-LM, and its hybrid SWA/GQA
architecture exercises all three layers.

### Baseline results (Gemma 4 E2B, CPU, x86_64, 16GB RAM)

`litert-lm` v0.10.1 with `gemma-4-E2B-it.litertlm` from
`litert-community/gemma-4-E2B-it-litert-lm`.

| Configuration | Prefill tok/s | Decode tok/s | TTFT | Init |
|:---|:---|:---|:---|:---|
| 256 prefill / 256 decode | 70.74 | 23.97 | 3.66s | 25.57s |
| 512 prefill / 128 decode | 125.98 | 20.78 | 4.11s | 24.78s |
| 1024 prefill / 128 decode | 271.62 | 22.46 | 3.81s | 23.18s |

Prefill scales superlinearly. Decode is stable at ~22 tok/s. TTFT is ~3.7-4.1s.

### Correctness (5/5 pass)

| Prompt | Expected | Result |
|:-------|:---------|:-------|
| "What is the capital of Japan?" | Tokyo | PASS |
| "What's the tallest building in the world?" | Burj Khalifa | PASS |
| "Write a Python function for Fibonacci" | `def fibonacci` | PASS |
| "List 5 differences: golden retriever vs labrador" | Numbered list | PASS |
| "Explain attention sinks in KV caches" | Technical accuracy | PASS |

### Run the evaluation yourself

```bash
chmod +x tools/eval/gemma4_aligned_eval.sh
./tools/eval/gemma4_aligned_eval.sh
```

Runs 8 correctness tests, 4 benchmarks, and a multi-turn simulation.
Set `BACKEND=gpu` for GPU, `MODEL_REPO=...` for a different model variant.

---

## What's Next

- **Tuning**: `prefetch_min_ratio`, per-builder strategy parameters, and
  `heat_score` calibration per workload profile.
- **New models**: each new release (Gemma, Qwen, Llama, etc.) needs a
  validated memory strategy profile.
- **Scaffold graduation**: `summarize_protected_tail` and `quarantine_merge`
  are deterministic placeholders; graduating them to real implementations is
  ongoing.
- **Literature**: ARKV (adaptive token states), KVSink (explicit sink
  preservation), and joint block encoding map onto the Native Cache layer's
  block model.

---

*Related docs: `PHASE_A_GATE.md`, `PHASE_B_GATE.md`, `PHASE_B_BOOTSTRAP.md`,
`PHASE_C_BOOTSTRAP.md`, `PHASE_C_CACHE_OPS_RFC_DRAFT.md`*
