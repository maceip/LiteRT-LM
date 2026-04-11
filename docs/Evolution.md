# Evolution

**A framework narrative for how LiteRT-LM's runtime memory policy evolves
across Phase A, Phase B, and Phase C — and why the shift is necessary for
privacy, user control, and ultimately performance.**

*A practical guide for engineers who just want to know what's happening
and why it matters.*

---

## Table of Contents

1. [Why This Document](#why-this-document)
2. [The Framework as A, B, and C](#the-framework-as-a-b-and-c)
3. [The Shift: Why It Is Necessary](#the-shift-why-it-is-necessary)
4. [Frontier Model Harrison Matrix](#frontier-model-harrison-matrix)
5. [System Architecture Reference](#system-architecture-reference)
6. [Phase A: Control-Plane Safety](#phase-a-control-plane-safety)
7. [Phase B: Predictive Prefetch Middleware](#phase-b-predictive-prefetch-middleware)
8. [Phase C: Native Cache Operations](#phase-c-native-cache-operations)
9. [The Golden Retriever: What Each Phase Brings Back](#the-golden-retriever-what-each-phase-brings-back)
10. [Where Do We Go from Here?](#where-do-we-go-from-here)

---

## Why This Document

Every phase of this architecture exists because there was a concrete problem
to solve.

This project did not arrive at its current architecture through careful
committee planning over years. It got here because real products — Chrome,
Chromebook Plus, Pixel Watch, the AI Edge Gallery — needed on-device LLM
inference that actually works under real constraints: memory budgets, thermal
limits, user privacy requirements, and hardware that ships in the millions.

Phase A solved the safety problem — making policy transitions deterministic.
Phase B solved the latency problem — making context shifts non-blocking.
Phase C solves the intelligence problem — giving the engine native awareness
of its own cache.

If you are reading this document, you are somewhere on that trajectory. This
document tells you where the framework has been, where it is, and where it
is going.

---

## The Framework as A, B, and C

The full framework decomposes into three phases. Each phase builds on the
guarantees of the one before it. None of them replaces the earlier phase — they
stack.

```
  THE THREE PHASES — LAYERED, NOT REPLACED
  ==========================================

  Phase C ── Native Cache Operations ────────────────────────────┐
  │  Engine-native KV surgery: Pin, EvictRange, Remap,           │
  │  Compact, SnapshotRestore. Capability-gated.                 │
  │  Falls back to Phase B on any failure.                       │
  ├──────────────────────────────────────────────────────────────-┤
  Phase B ── Predictive Prefetch Middleware ──────────────────────┤
  │  Background planning, precomputed replay packs,              │
  │  boundary-safe install, structured telemetry.                │
  │  Falls back to synchronous replay on any mismatch.           │
  ├──────────────────────────────────────────────────────────────-┤
  Phase A ── Control-Plane Safety ───────────────────────────────┤
  │  Safe-boundary queueing, atomic-turn enforcement,            │
  │  priority arbitration, transition notes,                     │
  │  version/compatibility gating.                               │
  │  The foundation. Always enforced. Never bypassed.            │
  └──────────────────────────────────────────────────────────────-┘
```

**Phase A** is the safety contract. It guarantees that runtime policy changes
never apply mid-turn, never corrupt active inference, and always respect
profile constraints.

**Phase B** is the performance layer. It adds background prefetch planning and
precomputed replay packs so that context-shift transitions — the expensive
moments where the runtime has to decide what to keep and what to discard — can
be computed ahead of time instead of blocking the user.

**Phase C** is the intelligence layer. It introduces engine-native cache
operations — real KV surgery — so the runtime can pin attention sinks, evict
ranges, remap blocks, and compact memory without full recompute. But it only
does this when the engine explicitly advertises capability support, and it falls
back to Phase B deterministic recompute on any failure.

### The fallback chain

```
  ┌────────────┐     capability     ┌────────────┐     always     ┌────────────┐
  │  Phase C   │ ──── missing? ───> │  Phase B   │ ── present ──> │  Phase A   │
  │  Native    │     or failure     │  Prefetch  │   and enforced │  Safety    │
  │  Cache Ops │                    │  Replay    │                │  Contract  │
  └────────────┘                    └────────────┘                └────────────┘
       │                                  │                             │
       │   On failure:                    │   On mismatch:              │
       │   rollback_unavailable           │   stale watermark           │
       │   internal_corruption            │   policy change             │
       │   unsupported_capability         │   history revision          │
       │                                  │                             │
       └──────── fall back to B ──────────┘──── always enforced ────────┘
```

This chain is not aspirational. It is the actual contract. Phase C never
strands a conversation in a partially shifted state. Phase B never blocks a
user turn on a planner thread. Phase A never allows a policy change mid-turn.

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

The phased architecture addresses this:

- **Phase A** ensures policy transitions are atomic and observable, so privacy
  auditing can attach to well-defined boundaries instead of racing against
  mid-turn state mutations.
- **Phase B** introduces replay packs with explicit retained-range metadata, so
  the runtime knows exactly which conversation segments are preserved and which
  are discarded — not by accident, but by policy.
- **Phase C** adds pin classes (`system_anchor`, `attention_sink`,
  `protected_tail`, `tool_state`, `ephemeral`) that make data-lifecycle
  semantics explicit at the cache-block level, enabling fine-grained retention
  and eviction policies that can align with user privacy preferences.

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

  After (Phased architecture):
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

Edge inference should serve the user, not the runtime. The phased architecture
gives users (and the applications built on top of LiteRT-LM) meaningful control
over how memory is managed:

- **Profile constraints** (`allow_runtime_tuning`, `safe_boundary`,
  `shadow_strategy`) let applications define policies that the runtime must
  respect. A medical application can require that system prompts are never
  evicted. A creative writing app can allow aggressive compaction. The runtime
  honors these constraints through the Phase A priority arbiter.

- **Strategy selection** is middleware-owned across all three phases. The engine
  never unilaterally decides what to keep and what to discard. Even in Phase C,
  where the engine has native cache-surgery capabilities, the middleware chooses
  the strategy and the engine executes it.

- **Observability** is built in from Phase A forward. Transition notes,
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
  │  Enforces: Phase A safety, Phase B freshness,           │
  │            Phase C capability gating                     │
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

The performance story across phases:

| Concern | Before Phases | Phase A | Phase B | Phase C |
|:---|:---|:---|:---|:---|
| Context shift | Full recompute | Safe transitions | Prefetch + install | Native KV surgery |
| Latency | Unbounded spike at threshold | Deterministic boundary | Background planning | Sub-linear evict/remap |
| Memory | Opaque blob management | Policy-constrained | Replay-pack metadata | Block-level accounting |
| Overhead | None (but also no intelligence) | Minimal policy checks | Planner thread cost | Capability discovery cost |

The key insight: **performance gains from Phase B and Phase C are only safe
because Phase A guarantees are in place.** Without atomic-turn enforcement, a
prefetch install could corrupt mid-turn inference. Without safe-boundary
queueing, a native cache operation could apply during active decode. The phases
are not independent optimizations — they are a safety-first performance stack.

```
  PERFORMANCE: LATENCY PROFILE ACROSS PHASES
  ============================================

  Context usage ──────────────────────────────────────>  100%
                                                 │
  No Phases:     ─────────────────────────────── SPIKE ──
                                                 │
  Phase A only:  ─────────────────────────────── spike ──
                                          (safe, same cost)
                                                 │
  Phase B:       ───────── plan ──── install ─── smooth ─
                      (background)  (boundary)   │
                                                 │
  Phase C:       ───────── plan ── native-op ─── minimal ─
                      (background)  (atomic)     │
                                                 │
```

---

## Frontier Model Harrison Matrix

*How the April 2026 frontier class manages memory — and what it means for
LiteRT-LM's phased architecture.*

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

The frontier matrix maps directly onto the phased architecture:

**Phase A relevance**: Every model in the matrix, regardless of cache strategy,
needs safe policy transitions. A Gemma 4 model running on a Pixel Watch with
128K context and a Qwen3 model running with 262K context both need atomic-turn
enforcement and boundary-safe policy changes. Phase A is universal.

**Phase B relevance**: Models with hybrid SWA (Gemma 4, Step 3.5 Flash) benefit
most from Phase B prefetch planning because their context-shift behavior is
predictable — local layers evict at a fixed window, so the prefetch planner can
anticipate exactly when shifts will occur. Models with MLA (DeepSeek V3.2) have
lower cache pressure, so Phase B triggers less frequently but remains the
fallback.

**Phase C relevance**: The Phase C block model (`block_id`, `token_span`,
`pin_class`, `heat_score`) is designed to accommodate all four cache-pressure
strategies:

- **GQA models** (Qwen3, Gemma 4): Blocks correspond to shared KV head groups.
  Pin/evict/remap operate on the reduced KV representation.
- **MLA models** (DeepSeek V3.2): Blocks correspond to latent vectors. The
  block model's `token_span` maps to the logical tokens that the latent
  represents, even though the physical storage is compressed.
- **Hybrid SWA models** (Gemma 4, Step 3.5 Flash): Blocks in local layers have
  bounded lifetime. The `heat_score` and eviction policies align with the
  sliding-window semantics — old blocks in local layers naturally have zero heat.
- **iRoPE models** (Llama 4): KV cache grows linearly, so Phase C eviction
  and compaction are most valuable. The `pin_class` system protects attention
  sinks that iRoPE's NoPE layers rely on for content-based attention.

```
  PHASE RELEVANCE BY CACHE STRATEGY
  ====================================

  Strategy              Phase A    Phase B         Phase C
  ────────              ───────    ───────         ───────
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

Three patterns emerge from the matrix that should inform the phased
architecture's roadmap:

1. **MoE dominance**: 8 of 10 open-weight models are MoE. Active parameter
   counts range from 4B to 37B. This means edge deployment is viable for models
   with hundreds of billions of total parameters — but only if the runtime can
   manage KV cache for the active parameter set efficiently. Phase C's
   block-level accounting is designed for this.

2. **Hybrid attention is the norm**: Gemma 4 and Step 3.5 Flash both use
   hybrid SWA/full-attention layouts. This creates two-tier cache behavior
   within a single model — local layers with bounded cache and global layers
   with unbounded cache. Phase B's prefetch planner and Phase C's
   per-block metadata are both designed to handle this heterogeneous behavior.

3. **Agentic use is universal**: Every model in the matrix emphasizes tool
   calling, function calling, or multi-agent orchestration. Agentic use means
   long-lived sessions with many tool-result boundaries — exactly the pattern
   that Phase A's boundary-safe policy transitions and Phase B's
   prefetch-at-boundary install are designed for.

---

## System Architecture Reference

The following diagram shows how the phased memory-policy system fits into the
broader LiteRT-LM runtime architecture. This is not the full runtime — it
focuses on the components relevant to context management and the phase
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
  │  │  │  Phase A      │  │  Phase B      │  │  Phase C               │  │  │
  │  │  │  Safety       │  │  Prefetch     │  │  Native Ops            │  │  │
  │  │  │              │  │              │  │                        │  │  │
  │  │  │ - boundary   │  │ - planner    │  │ - capability discovery │  │  │
  │  │  │   queueing   │  │ - replay     │  │ - CacheOpGroup exec   │  │  │
  │  │  │ - atomic     │  │   packs      │  │ - Pin / Evict / Remap │  │  │
  │  │  │   turn rule  │  │ - installer  │  │ - Compact / Snapshot  │  │  │
  │  │  │ - priority   │  │ - telemetry  │  │ - rollback envelope   │  │  │
  │  │  │   arbiter    │  │ - fallback   │  │ - fallback to B       │  │  │
  │  │  │ - transition │  │   to sync    │  │                        │  │  │
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
  │  │  Checkpoint   │  │  KV Cache (block model, Phase C) │               │
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
  2. Phase A: Is a turn active?
     ├── YES ──> Queue policy change, wait for boundary
     └── NO ───> Proceed
                  │
                  ▼
  3. Phase A: Priority arbitration
     │  Profile constraints > Runtime overrides > Hard limits
     │
     ▼
  4. Phase B: Is there a valid prefetch pack?
     ├── YES ──> Validate freshness
     │           ├── Fresh ──> Install at boundary (fast path)
     │           └── Stale ──> Discard, fall to sync replay
     └── NO ───> Synchronous replay (baseline path)
                  │
                  ▼
  5. Phase C: Does engine advertise native capability?
     ├── YES ──> Build CacheOpGroup
     │           ├── Success ──> Atomic commit (fastest path)
     │           └── Failure ──> Rollback, fall to Phase B recompute
     └── NO ───> Phase B handles it
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
  │  - When to plan (Phase B trigger ratio)                    │
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
  │  - CacheOpGroup atomic execution (Phase C)                 │
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

## Phase A: Control-Plane Safety

*Reference: `docs/PHASE_A_GATE.md`*

Phase A is the foundation. Every guarantee made in Phase B and Phase C depends
on Phase A being intact.

### What Phase A does

Phase A ensures that runtime memory-policy changes are **safe, deterministic,
and observable**. It does not optimize performance. It does not touch the KV
cache. It establishes the rules that everything else must follow.

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

### Why Phase A matters for privacy and control

Without Phase A, there is no way to guarantee that a privacy-sensitive
policy (e.g., "never evict the system prompt containing patient data
handling instructions") is respected during inference. A mid-turn policy
change could silently replace the active strategy with one that discards
protected content. Phase A makes this impossible.

---

## Phase B: Predictive Prefetch Middleware

*Reference: `docs/PHASE_B_BOOTSTRAP.md`, `docs/PHASE_B_GATE.md`*

Phase B is where performance improvements begin — without touching the engine's
internal KV representation.

### What Phase B does

When context usage approaches the shift threshold, Phase B starts background
planning: selecting what to retain, preparing replay data, and packaging it
into a precomputed replay pack. At the next safe boundary, if the pack is still
fresh, it installs instantly instead of performing a full synchronous replay.

### The prefetch lifecycle

```
  PHASE B — PREFETCH LIFECYCLE
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

Phase B introduces explicit builder identity for replay packs, which is
critical for telemetry, debugging, and Phase C compatibility:

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

Phase B emits structured telemetry across six dimensions: `profile_id`,
`strategy`, `builder_id`, `boundary`, `model_type`, and `reason_code`. These
dimensions are designed to align with Phase C native-path telemetry so that
performance comparisons across middleware and native paths are possible without
dimension translation.

---

## Phase C: Native Cache Operations

*Reference: `docs/PHASE_C_BOOTSTRAP.md`, `docs/PHASE_C_CACHE_OPS_RFC_DRAFT.md`*

Phase C is the frontier. It introduces engine-native KV surgery — real block-
level cache manipulation — behind explicit capability discovery.

### What Phase C does

Phase C gives the engine the ability to pin blocks, evict ranges, remap logical
spans, compact memory, and snapshot/restore cache state. These operations are
atomic at the operation-group level. They are only used when the engine
explicitly advertises support. On any failure, the system falls back to Phase B
deterministic recompute.

### The KV block model

```
  PHASE C — KV BLOCK MODEL
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

Phase C operations are gated by explicit capability flags. The engine declares
what it supports; the middleware checks before attempting any native operation.

```
  CAPABILITY GATING — DECISION TREE
  ====================================

  supports_kv_surgery?
  ├── false ──> Stay on Phase B. Do not attempt native ops.
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
  internal_cache_corruption ─────────┘    Run Phase B deterministic recompute.

  invalid_selector ──────────────────┐
  range_conflict ────────────────────┤──> Report to middleware.
  pinned_block_conflict ─────────────┤    Middleware may retry with
  position_semantics_violation ──────┤    adjusted parameters or
  summary_artifact_missing ──────────┤    fall back to Phase B.
  snapshot_not_found ────────────────┘
```

---

## The Golden Retriever: What Each Phase Brings Back

*Like a golden retriever returning with exactly what you threw — here is what
each phase fetches for the system.*

### Phase A fetches: Trust

```
  ┌──────────────────────────────────────────────────────────┐
  │  PHASE A — THE TRUST RETRIEVAL                           │
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

### Phase B fetches: Speed

```
  ┌──────────────────────────────────────────────────────────┐
  │  PHASE B — THE SPEED RETRIEVAL                           │
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

### Phase C fetches: Intelligence

```
  ┌──────────────────────────────────────────────────────────┐
  │  PHASE C — THE INTELLIGENCE RETRIEVAL                    │
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

### The full retrieval chain

```
  ┌────────┐      ┌────────┐      ┌──────────────┐
  │ Trust  │ ───> │ Speed  │ ───> │ Intelligence │
  │  (A)   │      │  (B)   │      │     (C)      │
  └────────┘      └────────┘      └──────────────┘
       │               │                  │
       │               │                  │
       ▼               ▼                  ▼
  Safety first.   Then fast.        Then smart.
  Always on.      Falls back        Falls back
                  to sync replay.   to Phase B.
```

---

## Where Do We Go from Here?

The three phases define the current trajectory. But a trajectory is not a
destination. Here is where the work continues.

### Tuning

The phased architecture exposes knobs that matter:

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

- **Heat score calibration** — Phase C's block metadata includes `heat_score`
  and `last_access_step`. The eviction and compaction policies that consume
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
  pressure profiles. The phase system is designed to absorb these differences
  through per-model strategy profiles, but the profiles themselves need to be
  created and validated for each new model.

- **New hardware** — NPU acceleration, GPU memory hierarchies, and edge-
  specific constraints (Pixel Watch vs. desktop GPU) all affect what cache
  operations are practical. Phase C's capability discovery is designed to
  handle this, but the capability definitions themselves will need to expand as
  new hardware surfaces become available.

- **Literature integration** — Papers like ARKV (adaptive token-state
  transitions), KVSink (explicit sink preservation), and joint block encoding
  describe techniques that map directly onto the Phase C block model. As these
  techniques mature, they should be integrated as new native operations or
  compaction strategies, not bolted on as special cases.

- **Scaffold graduation** — The Phase B scaffold builders
  (`summarize_protected_tail`, `quarantine_merge`) are explicitly identified as
  scaffolds. They are deterministic placeholders for future real semantic
  transforms. Graduating these scaffolds to full implementations — with actual
  summary generation and quarantine storage — is ongoing work that benefits from
  the explicit identity system already in place.

### Sharing

This is an open-source project. The phased architecture is designed to be
understandable, testable, and contributable.

- **Documentation as contract** — The phase gate documents (`PHASE_A_GATE.md`,
  `PHASE_B_GATE.md`, `PHASE_C_BOOTSTRAP.md`, `PHASE_C_CACHE_OPS_RFC_DRAFT.md`)
  are not aspirational design documents. They are contracts. Each one defines
  what must be true before the next phase begins. This document (`Evolution.md`)
  explains why those contracts exist and how they fit together.

- **Test evidence as proof** — Phase B's gate checklist requires four evidence
  buckets: unit coverage, concurrency coverage, long-session integration
  coverage, and performance comparison evidence. This pattern — evidence over
  assertion — should extend to Phase C and to community contributions.

- **Telemetry as shared language** — The structured telemetry dimensions
  (`profile_id`, `strategy`, `builder_id`, `boundary`, `model_type`,
  `reason_code`) provide a shared vocabulary for discussing runtime behavior.
  When contributors report performance results, regressions, or new strategies,
  they can use these dimensions to communicate precisely.

- **Fallback as safety net** — The universal fallback chain (Phase C fails to
  Phase B, Phase B fails to synchronous replay, Phase A is always enforced)
  means that contributions to Phase C cannot break Phase B, and contributions to
  Phase B cannot break Phase A. This makes the codebase safer to contribute to
  than a monolithic runtime where every change is load-bearing.

### The trajectory

```
  WHERE WE ARE AND WHERE WE'RE GOING
  =====================================

  Past                 Present              Future
  ────                 ───────              ──────

  Session/checkpoint   Phased memory        Adaptive, model-aware,
  model. Opaque.       policy. Layered.     hardware-responsive
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
*Framework: Phase A (Safety) → Phase B (Speed) → Phase C (Intelligence)*
*Audience: Engineers and contributors who want the full picture.*
