# Evolution: The Hurtle

**A framework narrative for how LiteRT-LM's runtime memory policy evolves
across Phase A, Phase B, and Phase C — and why the shift is necessary for
privacy, user control, and ultimately performance.**

*For the dumpy bulldogs among us who just want to know what's happening
and why it matters.*

---

## Table of Contents

1. [Why "Hurtle"](#why-hurtle)
2. [The Framework as A, B, and C](#the-framework-as-a-b-and-c)
3. [The Shift: Why It Is Necessary](#the-shift-why-it-is-necessary)
4. [System Architecture Reference](#system-architecture-reference)
5. [Phase A: Control-Plane Safety](#phase-a-control-plane-safety)
6. [Phase B: Predictive Prefetch Middleware](#phase-b-predictive-prefetch-middleware)
7. [Phase C: Native Cache Operations](#phase-c-native-cache-operations)
8. [The Golden Retriever: What Each Phase Brings Back](#the-golden-retriever-what-each-phase-brings-back)
9. [Where Do We Go from Here?](#where-do-we-go-from-here)

---

## Why "Hurtle"

A hurtle is not a graceful arc. It is forward momentum with weight behind it.

This project did not arrive at its current architecture through careful
committee planning over years. It got here because real products — Chrome,
Chromebook Plus, Pixel Watch, the AI Edge Gallery — needed on-device LLM
inference that actually works under real constraints: memory budgets, thermal
limits, user privacy requirements, and hardware that ships in the millions.

The phased evolution described here is the record of that momentum. Phase A
locked down safety. Phase B added speculative performance. Phase C opens the
door to native engine intelligence. Each phase hurtles forward, but each one
also makes sure the ground behind it is solid.

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

The hurtle continues. The ground behind us is solid. The ground ahead is being
built as we run.

---

*Document: `docs/Evolution.md`*
*Title: "The Hurtle"*
*Framework: Phase A (Safety) → Phase B (Speed) → Phase C (Intelligence)*
*Audience: Engineers, contributors, and dumpy bulldogs who want the full picture.*
