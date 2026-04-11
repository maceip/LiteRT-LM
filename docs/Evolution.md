# Evolution

**A framework narrative for how LiteRT-LM's runtime memory policy evolves
across three layers — Safety, Prefetch, and Native Cache — and why the shift
is necessary for privacy, user control, and ultimately performance.**

*A practical guide for engineers who just want to know what's happening
and why it matters.*

---

## Table of Contents

1. [Why This Document](#why-this-document)
2. [Three Layers](#three-layers)
3. [The Shift: Why It Is Necessary](#the-shift-why-it-is-necessary)
4. [Frontier Model Harrison Matrix](#frontier-model-harrison-matrix)
5. [System Architecture Reference](#system-architecture-reference)
6. [Safety Layer: Control-Plane Safety](#safety-layer-control-plane-safety)
7. [Prefetch Layer: Predictive Prefetch Middleware](#prefetch-layer-predictive-prefetch-middleware)
8. [Native Cache Layer: Engine-Native Operations](#native-cache-layer-engine-native-operations)
9. [The Golden Retriever: What Each Layer Delivers](#the-golden-retriever-what-each-layer-delivers)
10. [Aligned Evaluation: Gemma 4 and the Layered System](#aligned-evaluation-gemma-4-and-the-layered-system)
11. [Where Do We Go from Here?](#where-do-we-go-from-here)

---

## Why This Document

Every layer of this architecture exists because there was a concrete problem
to solve.

This project did not arrive at its current architecture through careful
committee planning over years. It got here because real products — Chrome,
Chromebook Plus, Pixel Watch, the AI Edge Gallery — needed on-device LLM
inference that actually works under real constraints: memory budgets, thermal
limits, user privacy requirements, and hardware that ships in the millions.

The Safety layer solved the safety problem — making policy transitions
deterministic. The Prefetch layer solved the latency problem — making context
shifts non-blocking. The Native Cache layer solves the intelligence problem —
giving the engine native awareness of its own cache.

If you are reading this document, you are somewhere on that trajectory. This
document tells you where the framework has been, where it is, and where it
is going.

---

## Three Layers

The full framework decomposes into three layers. Each layer builds on the
guarantees of the one below it. None of them replaces an earlier layer — they
stack.

```
  THREE LAYERS — STACKED, NOT REPLACED
  =======================================

  Native Cache ── Engine-Native Operations ──────────────────────┐
  │  Engine-native KV surgery: Pin, EvictRange, Remap,           │
  │  Compact, SnapshotRestore. Capability-gated.                 │
  │  Falls back to the Prefetch layer on any failure.            │
  ├──────────────────────────────────────────────────────────────-┤
  Prefetch ── Predictive Prefetch Middleware ─────────────────────┤
  │  Background planning, precomputed replay packs,              │
  │  boundary-safe install, structured telemetry.                │
  │  Falls back to synchronous replay on any mismatch.           │
  ├──────────────────────────────────────────────────────────────-┤
  Safety ── Control-Plane Safety ────────────────────────────────┤
  │  Safe-boundary queueing, atomic-turn enforcement,            │
  │  priority arbitration, transition notes,                     │
  │  version/compatibility gating.                               │
  │  The foundation. Always enforced. Never bypassed.            │
  └──────────────────────────────────────────────────────────────-┘
```

**The Safety layer** is the safety contract. It guarantees that runtime policy
changes never apply mid-turn, never corrupt active inference, and always
respect profile constraints.

**The Prefetch layer** is the performance layer. It adds background prefetch
planning and precomputed replay packs so that context-shift transitions — the
expensive moments where the runtime has to decide what to keep and what to
discard — can be computed ahead of time instead of blocking the user.

**The Native Cache layer** is the intelligence layer. It introduces engine-
native cache operations — real KV surgery — so the runtime can pin attention
sinks, evict ranges, remap blocks, and compact memory without full recompute.
But it only does this when the engine explicitly advertises capability support,
and it falls back to Prefetch-layer deterministic recompute on any failure.

### The fallback chain

```
  ┌────────────┐     capability     ┌────────────┐     always     ┌────────────┐
  │  Native    │ ──── missing? ───> │  Prefetch  │ ── present ──> │  Safety    │
  │  Cache     │     or failure     │  Replay    │   and enforced │  Contract  │
  │  Layer     │                    │  Layer     │                │  Layer     │
  └────────────┘                    └────────────┘                └────────────┘
       │                                  │                             │
       │   On failure:                    │   On mismatch:              │
       │   rollback_unavailable           │   stale watermark           │
       │   internal_corruption            │   policy change             │
       │   unsupported_capability         │   history revision          │
       │                                  │                             │
       └──── fall back to Prefetch ───────┘──── always enforced ────────┘
```

This chain is not aspirational. It is the actual contract. The Native Cache
layer never strands a conversation in a partially shifted state. The Prefetch
layer never blocks a user turn on a planner thread. The Safety layer never
allows a policy change mid-turn.

---

## The Shift: Why It Is Necessary

Three forces drive the evolution from a simple session/checkpoint model to a
layered memory-policy architecture. Each one matters independently. Together,
they make the shift non-optional.

### 1. Privacy

On-device inference exists because privacy matters. When a model runs on the
user's device, their data never leaves their hardware. But privacy is not just
about where the model runs — it is also about how the runtime manages context.

A naive runtime that sends entire conversation histories through recompute on
every context shift leaks temporal patterns through timing side channels,
creates unnecessary copies of sensitive context in memory, and makes it harder
to enforce data-lifecycle policies.

The layered architecture addresses this:

- **The Safety layer** ensures policy transitions are atomic and observable, so
  privacy auditing can attach to well-defined boundaries instead of racing
  against mid-turn state mutations.
- **The Prefetch layer** introduces replay packs with explicit retained-range
  metadata, so the runtime knows exactly which conversation segments are
  preserved and which are discarded — not by accident, but by policy.
- **The Native Cache layer** adds pin classes (`system_anchor`,
  `attention_sink`, `protected_tail`, `tool_state`, `ephemeral`) that make
  data-lifecycle semantics explicit at the cache-block level, enabling
  fine-grained retention and eviction policies that can align with user privacy
  preferences.

```
  PRIVACY: FROM OPAQUE TO EXPLICIT
  =================================

  Before (Session/Checkpoint model):
  ┌──────────────────────────────────────────────┐
  │  Opaque session blob                         │
  │  ┌────────────────────────────────────────┐  │
  │  │  System + User + Assistant + Tool ...  │  │
  │  │  (all mixed, no metadata, no policy)   │  │
  │  └────────────────────────────────────────┘  │
  │  On overflow: full recompute or truncate     │
  └──────────────────────────────────────────────┘

  After (Layered architecture):
  ┌──────────────────────────────────────────────┐
  │  Block-aware cache with explicit metadata    │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
  │  │system│ │ user │ │ asst │ │ tool │  ...   │
  │  │PINNED│ │ heat │ │ heat │ │PINNED│       │
  │  │sink  │ │ 0.3  │ │ 0.7  │ │state │       │
  │  └──────┘ └──────┘ └──────┘ └──────┘       │
  │  On overflow: policy-driven, auditable       │
  └──────────────────────────────────────────────┘
```

### 2. User Control

Edge inference should serve the user, not the runtime. The layered architecture
gives users (and the applications built on top of LiteRT-LM) meaningful control
over how memory is managed:

- **Profile constraints** (`allow_runtime_tuning`, `safe_boundary`,
  `shadow_strategy`) let applications define policies that the runtime must
  respect. A medical application can require that system prompts are never
  evicted. A creative writing app can allow aggressive compaction. The runtime
  honors these constraints through the Safety layer's priority arbiter.

- **Strategy selection** is middleware-owned across all three layers. The engine
  never unilaterally decides what to keep and what to discard. Even at the
  Native Cache layer, where the engine has native cache-surgery capabilities,
  the middleware chooses the strategy and the engine executes it.

- **Observability** is built in from the Safety layer forward. Transition notes,
  structured telemetry dimensions, builder identity, and reason codes make
  runtime behavior transparent to developers and auditable by platform owners.

```
  USER CONTROL: WHO DECIDES WHAT
  ===============================

  ┌─────────────────────────────────────────────────────────┐
  │                    APPLICATION LAYER                     │
  │  Sets: profile constraints, strategy preferences,       │
  │        privacy policies, retention requirements          │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    MIDDLEWARE LAYER                      │
  │  Owns: policy selection, fallback decisions,            │
  │        prefetch planning, builder identity               │
  │  Enforces: Safety rules, Prefetch freshness,             │
  │            Native Cache capability gating                │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │                     ENGINE LAYER                        │
  │  Owns: execution, KV allocation, block storage          │
  │  Executes: CacheOpGroups atomically                     │
  │  Reports: capabilities, failures, block metadata         │
  │  Never: unilaterally mutates policy                      │
  └─────────────────────────────────────────────────────────┘
```

### 3. Performance

This is the one that makes the headlines, but it only works if privacy and
control are already solid.

The performance story across layers:

| Concern | Before Layers | Safety | Prefetch | Native Cache |
|:---|:---|:---|:---|:---|
| Context shift | Full recompute | Safe transitions | Prefetch + install | Native KV surgery |
| Latency | Unbounded spike at threshold | Deterministic boundary | Background planning | Sub-linear evict/remap |
| Memory | Opaque blob management | Policy-constrained | Replay-pack metadata | Block-level accounting |
| Overhead | None (but also no intelligence) | Minimal policy checks | Planner thread cost | Capability discovery cost |

The key insight: **performance gains from the Prefetch and Native Cache layers
are only safe because the Safety layer guarantees are in place.** Without
atomic-turn enforcement, a prefetch install could corrupt mid-turn inference.
Without safe-boundary queueing, a native cache operation could apply during
active decode. The layers are not independent optimizations — they are a
safety-first performance stack.

```
  PERFORMANCE: LATENCY PROFILE ACROSS LAYERS
  ============================================

  Context usage ──────────────────────────────────────>  100%
                                                 │
  No layers:     ─────────────────────────────── SPIKE ──
                                                 │
  Safety only:   ─────────────────────────────── spike ──
                                          (safe, same cost)
                                                 │
  + Prefetch:    ───────── plan ──── install ─── smooth ─
                      (background)  (boundary)   │
                                                 │
  + Native Cache:───────── plan ── native-op ─── minimal ─
                      (background)  (atomic)     │
                                                 │
```

---

## Frontier Model Harrison Matrix

*How the April 2026 frontier class manages memory — and what it means for
LiteRT-LM's layered architecture.*

This section compares ten recent frontier model releases across the dimensions
that matter most for runtime memory policy: attention architecture, KV-cache
strategy, context length, cache-pressure relief mechanisms, and agentic/tool-use
posture. The comparison is grounded in primary-source release materials and
recent KV-cache research (see `docs/PHASE_C_CACHE_OPS_RFC_DRAFT.md` for the
full evidence basis).

### The verified frontier set

| # | Model | Lab | Date | Total Params | Active Params |
|:--|:------|:----|:-----|:-------------|:--------------|
| 1 | Gemma 4 (26B-A4B) | Google DeepMind | Apr 2026 | 26.1B | ~4B |
| 2 | Gemma 4 (31B dense) | Google DeepMind | Apr 2026 | 31B | 31B |
| 3 | Step 3.5 Flash | StepFun | Feb 2026 | 196B | ~11B |
| 4 | GLM-4.5 | Z.ai (Zhipu) | Jul 2025 | 355B | ~32B |
| 5 | GLM-4.5-Air | Z.ai (Zhipu) | Jul 2025 | 106B | ~12B |
| 6 | Qwen3-235B-A22B | Alibaba (Qwen) | Apr 2025 | 235B | ~22B |
| 7 | Muse Spark | Meta (MSL) | Apr 2026 | undisclosed | undisclosed |
| 8 | DeepSeek V3.2 | DeepSeek | Dec 2025 | 685B | ~37B |
| 9 | Llama 4 Scout | Meta | Jan 2026 | 109B | ~17B |
| 10 | Llama 4 Maverick | Meta | Jan 2026 | 400B | ~17B |

Extended set (API-only, architecture not publicly detailed):

| # | Model | Lab | Date | Notes |
|:--|:------|:----|:-----|:------|
| 11 | Claude Mythos Preview | Anthropic | Apr 2026 | Restricted to ~50 orgs via Project Glasswing |
| 12 | GPT-4.1 | OpenAI | Apr 2025 | 1M context, architecture undisclosed |
| 13 | Grok 4.1 | xAI | Nov 2025 | ~3T params (MoE), architecture largely undisclosed |

### Harrison Matrix: Attention Architecture

```
  FRONTIER ATTENTION ARCHITECTURES — APRIL 2026
  ================================================

  Model              Attention Type          KV Heads   Q Heads   Layers  Window
  ─────              ──────────────          ────────   ───────   ──────  ──────
  Gemma 4 26B-A4B    Hybrid SWA/Full         varies*    varies*   ~60     512-1K
                     (alternating layers)
  Gemma 4 31B        Hybrid SWA/Full         16 (GQA)   32        60      1,024
                     (alternating layers)
  Step 3.5 Flash     Hybrid SWA/Full         -**        -**       45      512
                     (3:1 SWA/Full ratio)
  GLM-4.5            MoE + undisclosed       n/a        n/a       n/a     n/a
  GLM-4.5-Air        MoE + undisclosed       n/a        n/a       n/a     n/a
  Qwen3-235B-A22B    GQA                     4          64        94      full
  Muse Spark         Undisclosed             n/a        n/a       n/a     n/a
  DeepSeek V3.2      MLA (latent compress)   n/a***     n/a***    n/a     full
  Llama 4 Scout      iRoPE (interleaved)     -**        -**       n/a     full****
  Llama 4 Maverick   iRoPE (interleaved)     -**        -**       n/a     full****

  *   Gemma 4 uses different head dims per layer type: local=256, global=512
  **  Specific head counts not publicly documented
  *** MLA replaces traditional Q/KV head model with latent compression
  **** iRoPE interleaves layers with and without positional encoding
```

### Harrison Matrix: KV-Cache Strategy

This is the core of the comparison — how each model family manages the memory
that grows with every token of context.

```
  FRONTIER KV-CACHE STRATEGIES — APRIL 2026
  ============================================

  ┌──────────────────┬──────────────────────────────────────────────────────┐
  │  STRATEGY        │  MODELS USING IT                                    │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │                  │                                                      │
  │  Hybrid SWA      │  Gemma 4 (all variants), Step 3.5 Flash             │
  │  (sliding +      │                                                      │
  │   full layers)   │  Local layers cache only last w tokens.             │
  │                  │  Global layers cache full sequence.                  │
  │                  │  Net effect: sublinear KV growth with ratio.        │
  │                  │                                                      │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │                  │                                                      │
  │  GQA             │  Qwen3-235B-A22B, Gemma 4 (within layers)           │
  │  (grouped-query  │                                                      │
  │   attention)     │  4-16 KV heads share across 32-64 Q heads.          │
  │                  │  Direct multiplicative KV-cache size reduction.     │
  │                  │  Qwen3: 4 KV heads / 64 Q heads = 93.75% savings.  │
  │                  │                                                      │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │                  │                                                      │
  │  MLA             │  DeepSeek V3.2                                       │
  │  (multi-latent   │                                                      │
  │   attention)     │  KV jointly compressed into low-rank latent.        │
  │                  │  ~98% per-token cache reduction vs standard MHA.    │
  │                  │  Decoupled RoPE: positional dims separate from      │
  │                  │  compressed dims to preserve weight absorption.     │
  │                  │  Only latent vector c_t^KV is cached, not full K/V. │
  │                  │                                                      │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │                  │                                                      │
  │  iRoPE           │  Llama 4 Scout, Llama 4 Maverick                    │
  │  (interleaved    │                                                      │
  │   RoPE)          │  Alternates layers WITH and WITHOUT positional      │
  │                  │  encoding. NoPE layers attend on content only.      │
  │                  │  Temperature scaling prevents attention degradation │
  │                  │  at extreme lengths (10M tokens for Scout).         │
  │                  │  KV cache still grows linearly — iRoPE improves    │
  │                  │  quality at length, not cache size.                 │
  │                  │                                                      │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │                  │                                                      │
  │  Context Cache   │  GLM-4.5, GLM-4.5-Air                               │
  │  (API-level      │                                                      │
  │   caching)       │  Serving-layer prefix caching for repeated prompts. │
  │                  │  Internal architecture not publicly documented.     │
  │                  │  128K context, 96K max output.                       │
  │                  │                                                      │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │                  │                                                      │
  │  Thought         │  Muse Spark                                          │
  │  Compression     │                                                      │
  │                  │  Compresses reasoning chains to use fewer tokens.   │
  │                  │  58M tokens/task vs Claude Opus 4.6's 157M.         │
  │                  │  Indirect cache pressure relief via reduced          │
  │                  │  reasoning length. Architecture not disclosed.      │
  │                  │                                                      │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │                  │                                                      │
  │  Prompt Cache    │  GPT-4.1 (API-level)                                 │
  │  (prefix reuse)  │                                                      │
  │                  │  Exact prefix match reuses KV tensors.              │
  │                  │  1024+ token minimum, 128-token granularity.        │
  │                  │  Up to 90% input cost reduction, 80% TTFT reduction.│
  │                  │  1M context window. Architecture not disclosed.     │
  │                  │                                                      │
  └──────────────────┴──────────────────────────────────────────────────────┘
```

### Harrison Matrix: Full Comparison

| Dimension | Gemma 4 | Step 3.5 Flash | GLM-4.5 | Qwen3-235B | Muse Spark | DeepSeek V3.2 | Llama 4 Scout | Llama 4 Maverick |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Architecture** | Dense + MoE variants | Sparse MoE | Sparse MoE | Sparse MoE | Undisclosed | Sparse MoE | Sparse MoE | Sparse MoE |
| **Total / Active** | 26-31B / 4-31B | 196B / 11B | 355B / 32B | 235B / 22B | n/a | 685B / 37B | 109B / 17B | 400B / 17B |
| **Context** | 128-256K | 256K | 128K | 262K-1M | n/a | 128K | 10M | 1M |
| **Attention** | Hybrid SWA/Full + GQA | Hybrid SWA/Full (3:1) | Undisclosed | GQA (4 KV / 64 Q) | Undisclosed | MLA (latent) | iRoPE | iRoPE |
| **KV reduction** | SWA local eviction + GQA head sharing | SWA local eviction (w=512) | API-level context cache | GQA 93.75% head reduction | Thought compression | MLA ~98% latent compression | None (linear growth) | None (linear growth) |
| **Positional** | RoPE (local θ=10K) + partial RoPE (global θ=1M) | Standard RoPE | Undisclosed | Standard RoPE | Undisclosed | Decoupled RoPE | iRoPE (interleaved NoPE) | iRoPE (interleaved NoPE) |
| **MTP** | No | MTP-3 (3-way) | No | Yes | No | Yes | No | No |
| **Tool/Function** | Native function calling | Massive tool orchestration | Native function calling | MCP + tool calling | Multi-agent orchestration | Function calling | Function calling | Function calling |
| **Thinking modes** | No | No | Yes (think/non-think) | Yes (think/non-think) | Visual chain-of-thought | No | No | No |
| **Edge target** | Yes (E2B/E4B) | No (cloud) | No (cloud/API) | No (cloud) | No (cloud) | No (cloud) | Yes (single H100) | No (4x A100) |
| **Open weights** | Yes | Yes | Yes | Yes | No | Yes | Yes | Yes |

### Cache pressure: a taxonomy

The matrix reveals four distinct strategies the frontier class uses to manage
KV-cache pressure. Understanding these strategies is critical for LiteRT-LM
because they determine what cache operations are meaningful at the edge.

```
  CACHE PRESSURE TAXONOMY — FOUR STRATEGIES
  ============================================

  1. ARCHITECTURAL COMPRESSION (build-time)
     ├── GQA: fewer KV heads, direct size reduction
     │   └── Qwen3: 4 KV / 64 Q = 16:1 sharing ratio
     │   └── Gemma 4: variable per layer type
     ├── MLA: low-rank latent projection, ~98% reduction
     │   └── DeepSeek V3.2: only latent c_t^KV cached
     └── Hybrid SWA: local layers discard old tokens
         └── Gemma 4: 50/60 layers are local (w=1024)
         └── Step 3.5 Flash: 3:1 SWA/Full (w=512)

  2. POSITIONAL INNOVATION (train-time)
     ├── iRoPE: NoPE layers + temperature scaling
     │   └── Llama 4: enables 10M context WITHOUT cache reduction
     ├── Decoupled RoPE: positional dims separated from latent
     │   └── DeepSeek V3.2: preserves weight absorption in MLA
     └── Partial RoPE: fractional positional encoding
         └── Gemma 4 global layers: partial=0.25, theta=1M

  3. INFERENCE-TIME OPTIMIZATION (serve-time)
     ├── Prefix caching: reuse KV for shared prompt prefixes
     │   └── GPT-4.1: exact prefix match, 1024+ tokens
     │   └── GLM-4.5: API-level context caching
     ├── PagedAttention: virtual memory for KV blocks
     │   └── Qwen3 + vLLM: 40% VRAM reduction
     └── Ring-buffer allocation: fixed-size cache rotation
         └── Gemma 4 SWA layers: ring-buffer for decode

  4. REASONING COMPRESSION (model-time)
     └── Thought compression: fewer tokens per reasoning step
         └── Muse Spark: 2.7x fewer tokens than Claude Opus 4.6
```

### What this means for LiteRT-LM

The frontier matrix maps directly onto the layered architecture:

**Safety layer relevance**: Every model in the matrix, regardless of cache
strategy, needs safe policy transitions. A Gemma 4 model running on a Pixel
Watch with 128K context and a Qwen3 model running with 262K context both need
atomic-turn enforcement and boundary-safe policy changes. The Safety layer is
universal.

**Prefetch layer relevance**: Models with hybrid SWA (Gemma 4, Step 3.5 Flash)
benefit most from Prefetch-layer planning because their context-shift behavior
is predictable — local layers evict at a fixed window, so the prefetch planner
can anticipate exactly when shifts will occur. Models with MLA (DeepSeek V3.2)
have lower cache pressure, so the Prefetch layer triggers less frequently but
remains the fallback.

**Native Cache layer relevance**: The Native Cache layer's block model
(`block_id`, `token_span`, `pin_class`, `heat_score`) is designed to
accommodate all four cache-pressure strategies:

- **GQA models** (Qwen3, Gemma 4): Blocks correspond to shared KV head groups.
  Pin/evict/remap operate on the reduced KV representation.
- **MLA models** (DeepSeek V3.2): Blocks correspond to latent vectors. The
  block model's `token_span` maps to the logical tokens that the latent
  represents, even though the physical storage is compressed.
- **Hybrid SWA models** (Gemma 4, Step 3.5 Flash): Blocks in local layers have
  bounded lifetime. The `heat_score` and eviction policies align with the
  sliding-window semantics — old blocks in local layers naturally have zero heat.
- **iRoPE models** (Llama 4): KV cache grows linearly, so Native Cache layer
  eviction and compaction are most valuable. The `pin_class` system protects
  attention sinks that iRoPE's NoPE layers rely on for content-based attention.

```
  LAYER RELEVANCE BY CACHE STRATEGY
  ====================================

  Strategy              Safety     Prefetch        Native Cache
  ────────              ──────     ────────        ────────────
  GQA                   Always     Prefetch at     Pin/Evict on reduced KV
                                   threshold       groups

  MLA                   Always     Rare trigger,   Latent-aware block ops
                                   still fallback  (compressed spans)

  Hybrid SWA            Always     Predictable     Window-aligned eviction,
                                   trigger point   bounded local blocks

  iRoPE (linear KV)     Always     Critical at     Highest value: evict +
                                   long context    compact relieve linear
                                                   growth

  Prompt/Context Cache   Always     Prefix-aware    Snapshot/Restore for
  (GPT-4.1, GLM-4.5)              planning         cached prefix boundaries

  Thought Compression    Always     Reduced cache   Standard block ops,
  (Muse Spark)                     pressure =       less frequent trigger
                                   fewer triggers
```

### Frontier convergence signals

Three patterns emerge from the matrix that should inform the layered
architecture's roadmap:

1. **MoE dominance**: 8 of 10 open-weight models are MoE. Active parameter
   counts range from 4B to 37B. This means edge deployment is viable for models
   with hundreds of billions of total parameters — but only if the runtime can
   manage KV cache for the active parameter set efficiently. The Native Cache
   layer's block-level accounting is designed for this.

2. **Hybrid attention is the norm**: Gemma 4 and Step 3.5 Flash both use
   hybrid SWA/full-attention layouts. This creates two-tier cache behavior
   within a single model — local layers with bounded cache and global layers
   with unbounded cache. The Prefetch layer's planner and the Native Cache
   layer's per-block metadata are both designed to handle this heterogeneous
   behavior.

3. **Agentic use is universal**: Every model in the matrix emphasizes tool
   calling, function calling, or multi-agent orchestration. Agentic use means
   long-lived sessions with many tool-result boundaries — exactly the pattern
   that the Safety layer's boundary-safe policy transitions and the Prefetch
   layer's prefetch-at-boundary install are designed for.

---

## System Architecture Reference

The following diagram shows how the layered memory-policy system fits into the
broader LiteRT-LM runtime architecture. This is not the full runtime — it
focuses on the components relevant to context management and the layered
evolution.

```
  LITERT-LM RUNTIME — MEMORY POLICY ARCHITECTURE
  =================================================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         APPLICATION / SDK                               │
  │  (Kotlin / Python / C++ / C API)                                       │
  │                                                                         │
  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
  │  │ Conversation │  │  Tool Use   │  │  Constrained │  │    Multi-    │ │
  │  │     API      │  │    API      │  │   Decoding   │  │   Modality   │ │
  │  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘ │
  └─────────┼─────────────────┼────────────────┼──────────────────┼────────┘
            │                 │                │                  │
            ▼                 ▼                ▼                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         CONVERSATION RUNTIME                            │
  │                                                                         │
  │  ┌───────────────────────────────────────────────────────────────────┐  │
  │  │                    MEMORY POLICY SUBSYSTEM                        │  │
  │  │                                                                   │  │
  │  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │  │
  │  │  │  Safety      │  │  Prefetch    │  │  Native Cache          │  │  │
  │  │  │  Layer       │  │  Layer       │  │  Layer                 │  │  │
  │  │  │              │  │              │  │                        │  │  │
  │  │  │ - boundary   │  │ - planner    │  │ - capability discovery │  │  │
  │  │  │   queueing   │  │ - replay     │  │ - CacheOpGroup exec   │  │  │
  │  │  │ - atomic     │  │   packs      │  │ - Pin / Evict / Remap │  │  │
  │  │  │   turn rule  │  │ - installer  │  │ - Compact / Snapshot  │  │  │
  │  │  │ - priority   │  │ - telemetry  │  │ - rollback envelope   │  │  │
  │  │  │   arbiter    │  │ - fallback   │  │ - fallback to         │  │  │
  │  │  │ - transition │  │   to sync    │  │   Prefetch            │  │  │
  │  │  │   notes      │  │   replay     │  │                        │  │  │
  │  │  └──────┬───────┘  └──────┬───────┘  └──────────┬─────────────┘  │  │
  │  │         │                 │                     │                │  │
  │  │         └─────────────────┼─────────────────────┘                │  │
  │  │                           │                                      │  │
  │  │                    POLICY ARBITER                                 │  │
  │  │              (profile > runtime > limits)                        │  │
  │  └───────────────────────────┼──────────────────────────────────────┘  │
  │                              │                                         │
  │  ┌───────────────────────────▼──────────────────────────────────────┐  │
  │  │                    CONTEXT SHIFT ENGINE                          │  │
  │  │  - strategy execution                                            │  │
  │  │  - retained slice computation                                    │  │
  │  │  - replay rendering                                              │  │
  │  │  - checkpoint management                                         │  │
  │  └───────────────────────────┬──────────────────────────────────────┘  │
  └──────────────────────────────┼─────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         ENGINE / EXECUTOR                               │
  │                                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
  │  │  RunPrefill   │  │  RunDecode   │  │    Clone     │                 │
  │  └──────────────┘  └──────────────┘  └──────────────┘                 │
  │  ┌──────────────┐  ┌──────────────────────────────────┐               │
  │  │  Checkpoint   │  │  KV Cache (block model,           │               │
  │  │  Save/Rewind  │  │  - block_id, token_span          │               │
  │  └──────────────┘  │  - lineage, pin_class             │               │
  │                     │  - heat_score, logical_role       │               │
  │                     └──────────────────────────────────┘               │
  └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    HARDWARE ACCELERATION LAYER                          │
  │  GPU / NPU / CPU backends                                              │
  └─────────────────────────────────────────────────────────────────────────┘
```

### Data flow during a context shift

```
  CONTEXT SHIFT — DATA FLOW
  ===========================

  1. Context usage exceeds threshold
     │
     ▼
  2. Safety layer: Is a turn active?
     ├── YES ──> Queue policy change, wait for boundary
     └── NO ───> Proceed
                  │
                  ▼
  3. Safety layer: Priority arbitration
     │  Profile constraints > Runtime overrides > Hard limits
     │
     ▼
  4. Prefetch layer: Is there a valid prefetch pack?
     ├── YES ──> Validate freshness
     │           ├── Fresh ──> Install at boundary (fast path)
     │           └── Stale ──> Discard, fall to sync replay
     └── NO ───> Synchronous replay (baseline path)
                  │
                  ▼
  5. Native Cache layer: Does engine advertise native capability?
     ├── YES ──> Build CacheOpGroup
     │           ├── Success ──> Atomic commit (fastest path)
     │           └── Failure ──> Rollback, fall to Prefetch recompute
     └── NO ───> Prefetch layer handles it
                  │
                  ▼
  6. Emit structured telemetry
     │  profile_id, strategy, builder_id, boundary,
     │  model_type, reason_code, outcome
     │
     ▼
  7. Resume inference at new context state
```

### Ownership boundaries

```
  OWNERSHIP BOUNDARIES
  =====================

  ┌────────────────────────────────────────────────────────────┐
  │              MIDDLEWARE OWNS                                │
  │                                                            │
  │  - Which strategy to use                                   │
  │  - When to plan (Prefetch trigger ratio)                   │
  │  - Whether to attempt native ops (capability check)        │
  │  - Fallback decisions                                      │
  │  - Telemetry dimensions and reason codes                   │
  │  - Builder identity and replay-pack lifecycle              │
  │  - Profile constraint enforcement                          │
  │                                                            │
  ├────────────────────────────────────────────────────────────┤
  │              ENGINE OWNS                                   │
  │                                                            │
  │  - KV allocation and block storage                         │
  │  - Prefill / Decode execution                              │
  │  - CacheOpGroup atomic execution (Native Cache)            │
  │  - Capability advertisement                                │
  │  - Rollback mechanics                                      │
  │  - Block metadata (id, span, lineage, heat, pin)           │
  │  - Checkpoint save / rewind                                │
  │                                                            │
  ├────────────────────────────────────────────────────────────┤
  │              NEITHER OWNS ALONE                            │
  │                                                            │
  │  - Context-shift outcome (middleware plans, engine runs)    │
  │  - Failure recovery (engine reports, middleware decides)    │
  │  - Telemetry (engine emits raw, middleware adds policy)     │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
```

---

## Safety Layer: Control-Plane Safety

*Reference: `docs/PHASE_A_GATE.md`*

The Safety layer is the foundation. Every guarantee made in the Prefetch and
Native Cache layers depends on the Safety layer being intact.

### What the Safety layer does

The Safety layer ensures that runtime memory-policy changes are **safe,
deterministic, and observable**. It does not optimize performance. It does not
touch the KV cache. It establishes the rules that everything else must follow.

### The five guarantees

| # | Guarantee | What it prevents |
|:--|:----------|:-----------------|
| 1 | Safe-boundary queueing | Policy changes applying at arbitrary moments |
| 2 | Atomic-turn enforcement | Policy changes during active prefill/decode |
| 3 | Priority arbitration | Runtime overrides bypassing profile constraints |
| 4 | Transition notes | Invisible policy transitions |
| 5 | Version/compatibility gating | Incompatible policies corrupting runtime state |

### Boundary model

```
  SAFE BOUNDARIES — WHERE POLICY CAN CHANGE
  ============================================

  ──── turn 1 ────── tool call ────── tool result ────── turn 2 ────
       │                                   │                  │
       │  POLICY LOCKED                    │  BOUNDARY        │  BOUNDARY
       │  (prefill/decode active)          │  (safe to apply) │  (safe to apply)
       │                                   │                  │
       ▼                                   ▼                  ▼
  ┌─────────┐                         ┌─────────┐       ┌─────────┐
  │ BLOCKED │                         │ ALLOWED │       │ ALLOWED │
  │ queued  │                         │ applied │       │ applied │
  └─────────┘                         └─────────┘       └─────────┘
```

### Why the Safety layer matters for privacy and control

Without the Safety layer, there is no way to guarantee that a privacy-sensitive
policy (e.g., "never evict the system prompt containing patient data
handling instructions") is respected during inference. A mid-turn policy
change could silently replace the active strategy with one that discards
protected content. The Safety layer makes this impossible.

---

## Prefetch Layer: Predictive Prefetch Middleware

*Reference: `docs/PHASE_B_BOOTSTRAP.md`, `docs/PHASE_B_GATE.md`*

The Prefetch layer is where performance improvements begin — without touching
the engine's internal KV representation.

### What the Prefetch layer does

When context usage approaches the shift threshold, the Prefetch layer starts
background planning: selecting what to retain, preparing replay data, and
packaging it into a precomputed replay pack. At the next safe boundary, if the
pack is still fresh, it installs instantly instead of performing a full
synchronous replay.

### The prefetch lifecycle

```
  PREFETCH LAYER — LIFECYCLE
  ==============================

  ┌──────────┐                                    Time ──────>
  │  Decode  │
  │  output  │
  └────┬─────┘
       │
       │  context_usage >= prefetch_min_ratio (e.g. 0.75)
       │
       ▼
  ┌──────────┐    background    ┌──────────┐    boundary    ┌──────────┐
  │ Planned  │ ──────────────>  │  Ready   │ ────────────>  │Installed │
  │          │    (async)       │          │   (validated)   │          │
  └──────────┘                  └────┬─────┘                └──────────┘
                                     │
                              stale? ├── policy changed ──> Discarded
                                     ├── history revised ─> Discarded
                                     └── superseded ──────> Discarded
```

### Builder identity

The Prefetch layer introduces explicit builder identity for replay packs, which
is critical for telemetry, debugging, and Native Cache layer compatibility:

| Builder ID | Behavior | Scaffold? |
|:-----------|:---------|:----------|
| `replay_recent` | Retains recent turns, replays from checkpoint | No |
| `drop_all_but_system` | Retains only system prompt | No |
| `summarize_protected_tail` | Marks summary anchor, retains tail | Yes |
| `quarantine_merge` | Merges quarantined segments | Yes |

Scaffold builders are explicitly identified in telemetry and artifacts. They
are deterministic and honest about their limitations — they do not pretend to
create real summaries or quarantine stores.

### Telemetry dimensions

The Prefetch layer emits structured telemetry across six dimensions:
`profile_id`, `strategy`, `builder_id`, `boundary`, `model_type`, and
`reason_code`. These dimensions are designed to align with Native Cache layer
telemetry so that performance comparisons across middleware and native paths are
possible without dimension translation.

---

## Native Cache Layer: Engine-Native Operations

*Reference: `docs/PHASE_C_BOOTSTRAP.md`, `docs/PHASE_C_CACHE_OPS_RFC_DRAFT.md`*

The Native Cache layer is the frontier. It introduces engine-native KV
surgery — real block-level cache manipulation — behind explicit capability
discovery.

### What the Native Cache layer does

The Native Cache layer gives the engine the ability to pin blocks, evict
ranges, remap logical spans, compact memory, and snapshot/restore cache state.
These operations are atomic at the operation-group level. They are only used
when the engine explicitly advertises support. On any failure, the system falls
back to Prefetch-layer deterministic recompute.

### The KV block model

```
  NATIVE CACHE LAYER — KV BLOCK MODEL
  ==========================

  Block identity:  (session_epoch, block_seqno)
  Token span:      [start_token, end_token)   (half-open, logical)

  ┌─────────────────────────────────────────────────────────────────┐
  │                     SESSION CACHE STATE                         │
  │                                                                 │
  │  block_id:     (1, 0)      (1, 1)      (1, 2)      (1, 3)     │
  │  token_span:   [0, 128)    [128, 384)  [384, 512)  [512, 640) │
  │  pin_class:    sink        ephemeral   ephemeral   tool_state  │
  │  logical_role: system      user        assistant   tool        │
  │  heat_score:   1.00        0.21        0.67        0.89        │
  │  is_pinned:    true        false       false       true        │
  │                                                                 │
  │  Lineage DAG:                                                   │
  │  (1,0) ← prefill                                               │
  │  (1,1) ← prefill                                               │
  │  (1,2) ← prefill                                               │
  │  (1,3) ← prefill                                               │
  └─────────────────────────────────────────────────────────────────┘

  After EvictRange([128, 384), strict) + Compact:

  ┌─────────────────────────────────────────────────────────────────┐
  │                     SESSION CACHE STATE                         │
  │                                                                 │
  │  block_id:     (1, 0)      (1, 4)            (1, 3)            │
  │  token_span:   [0, 128)    [128, 256)        [256, 384)        │
  │  pin_class:    sink        ephemeral         tool_state        │
  │  logical_role: system      assistant         tool              │
  │  heat_score:   1.00        0.67              0.89              │
  │  is_pinned:    true        false             true              │
  │                                                                 │
  │  Lineage DAG:                                                   │
  │  (1,0) ← prefill                                               │
  │  (1,4) ← compact [(1,2)]                                       │
  │  (1,3) ← prefill, remap                                        │
  └─────────────────────────────────────────────────────────────────┘
```

### Capability discovery

Native Cache layer operations are gated by explicit capability flags. The
engine declares what it supports; the middleware checks before attempting any
native operation.

```
  CAPABILITY GATING — DECISION TREE
  ====================================

  supports_kv_surgery?
  ├── false ──> Stay on Prefetch layer. Do not attempt native ops.
  │
  └── true ───> Check specific capabilities:
                │
                ├── supports_attention_sink_pinning?
                │   ├── false ──> No native compaction near sink blocks
                │   └── true ───> Pin(sink, attention_sink) is legal
                │
                ├── supports_range_evict?
                │   ├── false ──> No EvictRange
                │   └── true ───> EvictRange is legal
                │
                ├── supports_block_remap?
                │   ├── false ──> No Remap
                │   └── true ───> Remap is legal
                │
                ├── supports_compact?
                │   ├── false ──> No Compact
                │   └── true ───> Compact is legal
                │
                └── supports_snapshot_restore?
                    ├── false ──> No SnapshotRestore
                    └── true ───> SnapshotRestore is legal
```

### The operation vocabulary

| Operation | Purpose | Atomicity | Failure behavior |
|:----------|:--------|:----------|:-----------------|
| `Pin` | Protect blocks from eviction | Idempotent | Safe to retry |
| `EvictRange` | Remove KV state for a token interval | Group-atomic | Reject if pinned blocks conflict |
| `Remap` | Reassign logical span without re-prefill | Group-atomic | Fail if position semantics break |
| `Compact` | Reduce footprint via block replacement | Group-atomic | Fail if summary artifact missing |
| `SnapshotRestore` | Capture/restore full cache boundary | Group-atomic | Creates new session_epoch on restore |

### Failure codes and fallback

```
  FAILURE ──> FALLBACK DECISION
  ===============================

  unsupported_capability ────────────┐
  rollback_unavailable ──────────────┤──> Abandon native path.
  internal_cache_corruption ─────────┘    Run Prefetch deterministic recompute.

  invalid_selector ──────────────────┐
  range_conflict ────────────────────┤──> Report to middleware.
  pinned_block_conflict ─────────────┤    Middleware may retry with
  position_semantics_violation ──────┤    adjusted parameters or
  summary_artifact_missing ──────────┤    fall back to Prefetch.
  snapshot_not_found ────────────────┘
```

---

## The Golden Retriever: What Each Layer Delivers

*Like a golden retriever returning with exactly what you threw — here is what
each layer fetches for the system.*

### Safety layer delivers: Trust

```
  ┌──────────────────────────────────────────────────────────┐
  │  SAFETY LAYER — TRUST                                    │
  │                                                          │
  │  Throws:   "Make policy changes safe"                    │
  │  Returns:  Deterministic, auditable, atomic transitions  │
  │                                                          │
  │  Key artifacts:                                          │
  │  - Safe-boundary queue (tool_result, turn_boundary)      │
  │  - Atomic-turn enforcement (no mid-inference changes)    │
  │  - Priority arbiter (profile > runtime > limits)         │
  │  - Transition notes (observable policy shifts)           │
  │  - Version gating (reject before mutate)                 │
  │                                                          │
  │  Without this:  Nothing else can be trusted.             │
  └──────────────────────────────────────────────────────────┘
```

### Prefetch layer delivers: Speed

```
  ┌──────────────────────────────────────────────────────────┐
  │  PREFETCH LAYER — SPEED                                  │
  │                                                          │
  │  Throws:   "Make context shifts fast"                    │
  │  Returns:  Background planning, instant boundary install │
  │                                                          │
  │  Key artifacts:                                          │
  │  - Prefetch planner (async, bounded resources)           │
  │  - Replay packs (prebuilt, validated, identity-tagged)   │
  │  - Builder identity (replay_recent, drop_all_but_system, │
  │    summarize_protected_tail, quarantine_merge)            │
  │  - Structured telemetry (6 dimensions, 6 outcomes)       │
  │  - Scaffold honesty (explicit about what's real)         │
  │                                                          │
  │  Without this:  Every context shift blocks the user.     │
  └──────────────────────────────────────────────────────────┘
```

### Native Cache layer delivers: Intelligence

```
  ┌──────────────────────────────────────────────────────────┐
  │  NATIVE CACHE LAYER — INTELLIGENCE                       │
  │                                                          │
  │  Throws:   "Make the engine understand its own cache"    │
  │  Returns:  Block-level awareness, native KV surgery      │
  │                                                          │
  │  Key artifacts:                                          │
  │  - KV block model (identity, span, lineage, pin, heat)  │
  │  - Native ops (Pin, EvictRange, Remap, Compact,          │
  │    SnapshotRestore)                                      │
  │  - Capability discovery (explicit, versioned, immutable) │
  │  - CacheOpGroup atomicity (all-or-nothing commits)       │
  │  - Failure vocabulary (9 codes, 3 trigger fallback)      │
  │  - Rollback envelope (pre-state, touched set, viability) │
  │                                                          │
  │  Without this:  The engine is blind to what it holds.    │
  └──────────────────────────────────────────────────────────┘
```

### The full delivery chain

```
  ┌────────┐      ┌──────────┐      ┌──────────────┐
  │ Trust  │ ───> │  Speed   │ ───> │ Intelligence │
  │(Safety)│      │(Prefetch)│      │(Native Cache)│
  └────────┘      └──────────┘      └──────────────┘
       │               │                  │
       │               │                  │
       ▼               ▼                  ▼
  Safety first.   Then fast.        Then smart.
  Always on.      Falls back        Falls back
                  to sync replay.   to Prefetch.
```

---

## Aligned Evaluation: Gemma 4 and the Layered System

Gemma 4 is the primary evaluation target for the layered architecture because
it is the model this runtime is built for. It ships inside Google products
(Chrome, Chromebook Plus, Pixel Watch), it runs on edge hardware via LiteRT-LM,
and its architecture — hybrid sliding-window/full attention, GQA, MoE — is
exactly the workload the three layers were designed to serve.

This section defines how to evaluate the layered system against Gemma 4 end to
end: what to measure, how to measure it, and what alignment looks like.

### Why Gemma 4 is the alignment target

Gemma 4 exercises every layer of the system simultaneously:

```
  GEMMA 4 — LAYER ALIGNMENT MAP
  ================================

  Gemma 4 Feature             Layer Exercised       Why It Matters
  ───────────────             ───────────────       ──────────────
  Hybrid SWA/Full attention   Native Cache          Local layers have bounded
  (50 SWA / 10 Full layers)                         cache; global layers grow.
                                                    Block metadata must track
                                                    both behaviors per-layer.

  GQA (16 KV / 32 Q heads)   Native Cache          Blocks map to shared KV
                                                    head groups. Pin/evict ops
                                                    operate on the reduced
                                                    representation.

  128-256K context window     Prefetch              Context-shift threshold
                                                    triggers prefetch planning.
                                                    Hybrid SWA makes the trigger
                                                    point predictable.

  Native function calling     Safety                Tool-result boundaries are
  (code_fence_start/end,                            safe policy transition
  constraint_mode)                                  points. Atomic-turn rule
                                                    prevents mid-call changes.

  Vision + Audio modality     Safety + Prefetch     Multimodal tokens inflate
  (start_of_image_token,                            context faster. Prefetch
  start_of_audio_token,                             trigger ratio must account
  patch_width/height)                               for modality overhead.

  Edge deployment             All three             E2B/E4B variants target
  (E2B, E4B, 26B-A4B)                              phones, watches, RPi.
                                                    Memory constraints make
                                                    all three layers critical.
```

### What to measure

Aligned evaluation measures the layered system as a whole, not individual
layers in isolation. The evaluation covers three dimensions: correctness,
performance, and policy compliance.

#### Correctness

Does the output match regardless of which layer handles the context shift?

```
  CORRECTNESS EVALUATION
  ========================

  For each test prompt P and Gemma 4 variant V:

  1. Baseline run (no context shift):
     Engine → Conversation → SendMessage(P) → Response R_baseline

  2. Safety-only run (context shift at threshold, sync replay):
     Engine → Conversation → fill context to trigger_ratio →
     SendMessage(P) → context shift (sync replay) →
     Response R_safety

  3. Prefetch run (context shift with prefetch pack install):
     Engine → Conversation → fill context to prefetch_min_ratio →
     prefetch plans in background → fill to trigger_ratio →
     boundary install → SendMessage(P) → Response R_prefetch

  4. Native Cache run (when engine advertises capability):
     Engine → Conversation → fill context →
     CacheOpGroup(EvictRange + Compact) → SendMessage(P) →
     Response R_native

  Alignment check:
     R_baseline ≈ R_safety ≈ R_prefetch ≈ R_native
     (semantic equivalence, not token-identical)
```

#### Performance

Where does the latency go during context shifts?

| Metric | Measurement point | Expected behavior |
|:---|:---|:---|
| Time-to-first-token (TTFT) | `SendMessage` → first token callback | Prefetch and Native Cache should reduce TTFT vs Safety-only during context shifts |
| Context-shift latency | Boundary detection → resumed inference | Native Cache < Prefetch < Safety-only |
| Prefetch hit rate | Prefetch pack installed vs discarded | Higher is better; Gemma 4's predictable SWA trigger should yield high hit rates |
| Tokens per second (decode) | Steady-state decode throughput | Should not regress across layers |
| Peak memory | Max RSS during inference | Native Cache eviction should reduce peak vs full-recompute baseline |

#### Policy compliance

Does the system honor its own contracts during Gemma 4 inference?

| Policy | Evaluation method |
|:---|:---|
| Atomic-turn enforcement | Inject policy change during active decode; verify it queues, not applies |
| Boundary-safe transitions | Verify policy applies only at `tool_result` or `turn_boundary` |
| Priority arbitration | Set `allow_runtime_tuning=false` in profile; verify override is rejected |
| Fallback correctness | Disable native capabilities; verify Prefetch path produces equivalent output |
| Telemetry completeness | Verify all 6 dimensions emitted (`profile_id`, `strategy`, `builder_id`, `boundary`, `model_type`, `reason_code`) |

### How to evaluate

The evaluation uses the existing infrastructure in the repository, extended
with Gemma 4-specific test data and memory policy profiles.

```
  EVALUATION PIPELINE — GEMMA 4
  ================================

  ┌─────────────────────────────────────────────────────────────────┐
  │  INPUT                                                         │
  │                                                                │
  │  Model:    gemma-4-E2B-it.litertlm (or E4B, 26B-A4B, 31B)    │
  │  Profile:  memory_policy_gemma4.yaml                           │
  │  Tests:    test_data/test_gemma4_aligned.json                  │
  └────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ENGINE SETUP                                                  │
  │                                                                │
  │  ModelAssets::Create("gemma-4-E2B-it.litertlm")               │
  │  EngineSettings::CreateDefault(assets, backend)                │
  │       → Gemma4 proto auto-detected (has_gemma4())              │
  │       → delegate clustering disabled                           │
  │       → Gemma4DataProcessor created by factory                 │
  │  EngineFactory::CreateAny(settings)                            │
  │  Conversation::Create(engine, config)                          │
  │       → memory policy loaded from YAML profile                 │
  └────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  TEST EXECUTION                                                │
  │                                                                │
  │  For each test case in test_gemma4_aligned.json:               │
  │                                                                │
  │  1. Fill context with multi-turn conversation                  │
  │     (system prompt → user turns → assistant turns → tool calls)│
  │                                                                │
  │  2. Reach context_shift.trigger_ratio (e.g. 0.9)              │
  │                                                                │
  │  3. Send evaluation prompt                                     │
  │                                                                │
  │  4. Record:                                                    │
  │     - Response text (for correctness comparison)               │
  │     - BenchmarkInfo (TTFT, tokens/sec, shift latency)          │
  │     - Telemetry dimensions (profile_id, strategy, etc.)        │
  │     - Policy events (queued changes, applied changes, rejects) │
  │                                                                │
  │  5. Repeat with different layer configurations:                │
  │     - Safety only (prefetch disabled, native disabled)         │
  │     - Safety + Prefetch (native disabled)                      │
  │     - Safety + Prefetch + Native Cache (when available)        │
  └────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ALIGNMENT COMPARISON                                          │
  │                                                                │
  │  Compare across layer configurations:                          │
  │  - Semantic equivalence of responses                           │
  │  - Latency improvement ratios                                  │
  │  - Memory reduction ratios                                     │
  │  - Policy compliance (zero violations expected)                │
  │  - Telemetry completeness (all dimensions present)             │
  └─────────────────────────────────────────────────────────────────┘
```

### Gemma 4 memory policy profile

The evaluation requires a Gemma 4-specific memory policy profile. This profile
is tuned for Gemma 4's hybrid SWA architecture:

```yaml
  # memory_policy_gemma4.yaml
  #
  # Aligned evaluation profile for Gemma 4 variants.
  # Tuned for hybrid SWA/Full attention with predictable
  # local-layer cache eviction.

  version: 1
  profile_id: gemma4-aligned-eval
  strategy: replay_recent

  context_shift:
    enabled: true
    trigger_ratio: 0.85
    retain_recent_messages: 8
    target_ratio: 0.7
    reset_on_exhaustion: true
    shift_strategy: replay_recent

  prefetch:
    enabled: true
    min_ratio: 0.75
    shadow_mode: false

  constraints:
    allow_runtime_tuning: true
    safe_boundary: turn_boundary
    emit_transition_note: true
```

Key tuning decisions:

- **`trigger_ratio: 0.85`** — Gemma 4's hybrid SWA means local layers evict
  at a fixed window (512-1024 tokens), so the effective context pressure comes
  from global layers. Setting the trigger at 0.85 gives the Prefetch layer time
  to plan before the global layers saturate.

- **`prefetch.min_ratio: 0.75`** — Start prefetch planning at 75% context
  usage, giving 10% headroom before the trigger fires at 85%. For Gemma 4's
  predictable SWA behavior, this should yield high prefetch hit rates.

- **`retain_recent_messages: 8`** — Retain enough recent turns to preserve
  tool-call chains and function-response context, which Gemma 4's native
  function calling depends on.

- **`safe_boundary: turn_boundary`** — Apply policy changes at turn boundaries
  only. This aligns with Gemma 4's function-calling flow where tool results
  arrive as distinct turns.

### Gemma 4 test data structure

The evaluation test data extends the existing E2E sanity check pattern
(`tools/test/test_data/test_e2e_sanity_checks.json`) with Gemma 4-specific
cases that exercise the layered system:

```json
  {
    "test_gemma4_aligned": [
      {
        "id": "gemma4_basic_correctness",
        "prompt": "What is the capital of Japan?",
        "response": "Tokyo",
        "notes": "Baseline correctness after context shift"
      },
      {
        "id": "gemma4_tool_call_boundary",
        "prompt": "What is the weather in Tokyo?",
        "response": "tool_call|function_call",
        "notes": "Verify tool call survives context shift at boundary"
      },
      {
        "id": "gemma4_multi_turn_retention",
        "prompt": "Summarize our conversation so far",
        "response": ".*",
        "notes": "Verify retained messages are coherent post-shift"
      },
      {
        "id": "gemma4_system_prompt_pinned",
        "prompt": "What are your instructions?",
        "response": ".*",
        "notes": "System prompt must survive all context shift strategies"
      }
    ]
  }
```

### Alignment criteria

The evaluation succeeds — the system is aligned — when all of the following
are true for Gemma 4:

```
  ALIGNMENT CRITERIA
  ====================

  ┌─────────────────────────────────────────────────────────────────┐
  │  CORRECTNESS                                                   │
  │                                                                │
  │  ✓ Responses are semantically equivalent across all layer      │
  │    configurations (Safety-only, +Prefetch, +Native Cache)      │
  │  ✓ Tool calls survive context shifts at boundaries             │
  │  ✓ System prompt is never evicted                              │
  │  ✓ Retained messages maintain coherence post-shift             │
  ├─────────────────────────────────────────────────────────────────┤
  │  PERFORMANCE                                                   │
  │                                                                │
  │  ✓ Prefetch install reduces TTFT vs Safety-only baseline       │
  │  ✓ Prefetch hit rate > 80% for Gemma 4 SWA workloads          │
  │  ✓ Decode throughput does not regress across configurations    │
  │  ✓ Peak memory is bounded by context_shift.target_ratio        │
  ├─────────────────────────────────────────────────────────────────┤
  │  POLICY                                                        │
  │                                                                │
  │  ✓ Zero mid-turn policy applications                           │
  │  ✓ All policy changes applied at configured safe_boundary      │
  │  ✓ Priority arbiter rejects overrides when disallowed          │
  │  ✓ Fallback from Native Cache → Prefetch produces equivalent  │
  │    output                                                      │
  │  ✓ All 6 telemetry dimensions present on every context shift   │
  └─────────────────────────────────────────────────────────────────┘
```

### How this maps to the existing codebase

| Evaluation step | Existing infrastructure | What exists today |
|:---|:---|:---|
| Model loading | `ModelAssets::Create` → `EngineSettings` → `EngineFactory` | Gemma 4 auto-detection via `has_gemma4()` in `engine_settings.cc` |
| Data processing | `Gemma4DataProcessor` via `ModelDataProcessorFactory` | Full implementation with unit tests in `gemma4_data_processor_test.cc` |
| Prompt rendering | `model_type_utils.cc` default Jinja template for `kGemma4` | Template registered, tested in `model_type_utils_test.cc` |
| Memory policy | `ConversationConfig` with YAML profile | Existing `memory_policy_16.yaml` pattern, extensible per model |
| E2E execution | `litert_lm_main.cc` / `litert_lm_advanced_main.cc` | `--benchmark` flag, `BenchmarkInfo` output, metric proto export |
| Sanity checks | `tools/test/test_e2e_sanity_checks.py` with `conftest.py` | JSON-driven, model-agnostic, extensible with new test data files |
| Benchmark comparison | `BenchmarkInfo` (TTFT, tokens/sec) | Available via C++, Python CLI (`litert-lm benchmark`), and Kotlin JNI |

The aligned evaluation does not require new infrastructure. It requires a
Gemma 4-specific profile, Gemma 4-specific test data, and a test harness that
runs the same prompts across layer configurations and compares results.

---

## Where Do We Go from Here?

The three layers define the current trajectory. But a trajectory is not a
destination. Here is where the work continues.

### Tuning

The layered architecture exposes knobs that matter:

- **`prefetch_min_ratio`** — When does background planning start? Set it too
  low and you waste compute on plans that are never needed. Set it too high and
  the plan is not ready when the boundary arrives. The right value depends on
  the model, the hardware, and the use case. Tuning this ratio per-deployment
  is where real-world performance gains will come from.

- **Strategy parameters** — Each builder (`replay_recent`,
  `drop_all_but_system`, etc.) has implicit parameters: how many recent turns
  to retain, whether to protect tool-state blocks, how aggressively to compact.
  These parameters should become explicit, profile-configurable, and
  observable via telemetry.

- **Heat score calibration** — The Native Cache layer's block metadata includes
  `heat_score` and `last_access_step`. The eviction and compaction policies that consume
  these scores need calibration against real workloads. A coding assistant has
  different access patterns than a medical triage bot. The heat model should be
  tunable per-profile.

- **Capability-aware strategy selection** — When the engine advertises partial
  capabilities (e.g., `supports_range_evict=true` but
  `supports_compact=false`), the middleware's strategy selection should
  automatically adapt. This is not yet fully implemented. The goal is a strategy
  selector that consults capability flags and produces the best plan for the
  available operations, without manual per-engine configuration.

### Updating

The codebase is alive. Models change. Hardware changes. The KV-cache literature
is moving fast.

- **New models** — Every new model release (Gemma 4, Step 3.5 Flash, Qwen3,
  etc.) brings different attention patterns, context lengths, and cache
  pressure profiles. The layered system is designed to absorb these differences
  through per-model strategy profiles, but the profiles themselves need to be
  created and validated for each new model.

- **New hardware** — NPU acceleration, GPU memory hierarchies, and edge-
  specific constraints (Pixel Watch vs. desktop GPU) all affect what cache
  operations are practical. The Native Cache layer's capability discovery is
  designed to handle this, but the capability definitions themselves will need
  to expand as new hardware surfaces become available.

- **Literature integration** — Papers like ARKV (adaptive token-state
  transitions), KVSink (explicit sink preservation), and joint block encoding
  describe techniques that map directly onto the Native Cache layer's block
  model. As these
  techniques mature, they should be integrated as new native operations or
  compaction strategies, not bolted on as special cases.

- **Scaffold graduation** — The Prefetch layer's scaffold builders
  (`summarize_protected_tail`, `quarantine_merge`) are explicitly identified as
  scaffolds. They are deterministic placeholders for future real semantic
  transforms. Graduating these scaffolds to full implementations — with actual
  summary generation and quarantine storage — is ongoing work that benefits from
  the explicit identity system already in place.

### Sharing

This is an open-source project. The layered architecture is designed to be
understandable, testable, and contributable.

- **Documentation as contract** — The layer gate documents
  (`PHASE_A_GATE.md`, `PHASE_B_GATE.md`, `PHASE_C_BOOTSTRAP.md`,
  `PHASE_C_CACHE_OPS_RFC_DRAFT.md`) are not aspirational design documents.
  They are contracts. Each one defines what must be true before the next layer
  begins. This document (`Evolution.md`) explains why those contracts exist and
  how they fit together.

- **Test evidence as proof** — The Prefetch layer's gate checklist requires
  four evidence buckets: unit coverage, concurrency coverage, long-session
  integration coverage, and performance comparison evidence. This pattern —
  evidence over assertion — should extend to the Native Cache layer and to
  community contributions.

- **Telemetry as shared language** — The structured telemetry dimensions
  (`profile_id`, `strategy`, `builder_id`, `boundary`, `model_type`,
  `reason_code`) provide a shared vocabulary for discussing runtime behavior.
  When contributors report performance results, regressions, or new strategies,
  they can use these dimensions to communicate precisely.

- **Fallback as safety net** — The universal fallback chain (Native Cache fails
  to Prefetch, Prefetch fails to synchronous replay, Safety is always enforced)
  means that contributions to the Native Cache layer cannot break the Prefetch
  layer, and contributions to the Prefetch layer cannot break the Safety layer.
  This makes the codebase safer to contribute to than a monolithic runtime where
  every change is load-bearing.

### The trajectory

```
  WHERE WE ARE AND WHERE WE'RE GOING
  =====================================

  Past                 Present              Future
  ────                 ───────              ──────

  Session/checkpoint   Layered memory        Adaptive, model-aware,
  model. Opaque.       policy. Stacked.     hardware-responsive
  Full recompute       Safety + speed +     cache intelligence.
  on overflow.         native ops.          Per-deployment tuning.
                                            Graduated scaffolds.
                                            Community strategies.

  ┌─────┐             ┌─────┐              ┌─────┐
  │  ●  │ ──────────> │  ●  │ ──────────>  │  ●  │
  └─────┘             └─────┘              └─────┘
  "It works"          "It works safely,    "It works safely,
                       fast, and smart"     and it learns"
```

The next problem is ahead. The ground behind us is solid. The ground ahead is
being built as we run.

---

*Document: `docs/Evolution.md`*
*Title: "Evolution"*
*Framework: Safety Layer → Prefetch Layer → Native Cache Layer*
*Audience: Engineers and contributors who want the full picture.*
