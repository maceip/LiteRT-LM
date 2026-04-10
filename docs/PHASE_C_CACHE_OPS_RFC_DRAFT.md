# Phase C Cache Operations RFC Draft

This document defines a Phase C cache-operations contract grounded in:

1. primary-source frontier model release materials from 2025-2026,
2. primary-source 2025-2026 KV-cache research,
3. the current `LiteRT-LM` engine/session abstraction inherited from upstream
   `google-ai-edge/LiteRT-LM`.

The goal is not to claim that frontier labs publicly disclose all KV internals.
They generally do not. Instead, this document distinguishes:

- what primary sources explicitly say,
- what recent literature suggests is becoming standard practice,
- what contract best fits this repository's current architecture.

## Evidence basis

### Current repo / upstream constraints

Current `Engine::Session` in this repo is still a black-box interface centered on:

- `RunPrefill` / `RunPrefillAsync`
- `RunDecode` / `RunDecodeAsync`
- `Clone` / `CloneAsync`
- `SaveCheckpoint`
- `RewindToCheckpoint`
- `GetCurrentStep`

There is no native KV-surgery capability surface in the current engine API.

### Verified recent frontier release set

The following list is the proposed verified set of ten recent frontier releases,
based on primary sources available during this task. These are not all equal in
"frontier" strength, but they form a defensible current set spanning the major
labs and open/frontier hybrids:

1. Anthropic Claude Mythos Preview (Apr 2026)
2. Google Gemini 3.1 Pro (Feb 2026)
3. StepFun Step 3.5 Flash (Feb 2026)
4. xAI Grok 4.1 (Nov 2025)
5. DeepSeek V3.2 (Dec 2025)
6. Z.ai GLM-4.5 (Jul 2025)
7. Z.ai GLM-4.5-Air (Jul 2025)
8. Meta Llama 4 Maverick / Scout (Apr 2025)
9. OpenAI GPT-4.1 (Apr 2025)
10. Qwen3-235B-A22B (Apr 2025)

Additional strong-but-not-in-top-10 candidate:

- OpenAI GPT-4.5 (Feb 2025), which was superseded by GPT-4.1 for API use.

### What primary release sources publicly reveal about cache operations

Across those releases, public materials consistently reveal:

- context length,
- attention style (e.g. sliding window, interleaved attention),
- grouped-query or multi-query style hints,
- speculative decoding / MTP,
- reasoning / tool-use modes,
- API availability and deployment framework support.

Across those same releases, public materials generally do **not** reveal:

- explicit KV block model definitions,
- native operation vocabulary like `Pin`, `Remap`, or `EvictRange`,
- formal rollback guarantees,
- explicit engine capability-discovery schemas.

This means the contract below must be inferred from:

- public architecture clues,
- serving-framework norms,
- recent KV-cache papers,
- this repo's current black-box session model.

## Synthesis from frontier releases

### Strong observed trends

The recent frontier releases suggest the following durable design pressures:

1. **Long context is routine, not exceptional**
   - Meta Llama 4 Scout advertises 10M context and interleaved attention.
   - OpenAI GPT-4.1 exposes 1M-token context.
   - Gemini 3.1 Pro is framed around complex long-form synthesis.
   - Step 3.5 Flash uses a 256K hybrid SWA/full-attention layout.
   - Qwen3 and GLM-4.5 public docs emphasize long context and agentic use.

2. **Agentic/tool-use operation is first-class**
   - OpenAI emphasizes Responses API / agentic applications.
   - GLM-4.5 emphasizes native function calling and agentic coding.
   - Qwen3 emphasizes MCP and tool calling.
   - Step 3.5 Flash emphasizes massive tool orchestration and agent loops.

3. **Hybrid thinking / non-thinking modes are becoming common**
   - GLM-4.5 and Qwen3 explicitly expose thinking/non-thinking modes.
   - Grok 4.1 distinguishes thinking / non-thinking serving modes.

4. **Efficient long-context serving depends on constrained attention or cache pressure relief**
   - Step 3.5 Flash: 3:1 sliding-window/full-attention ratio.
   - Llama 4 Scout: interleaved attention architecture.
   - Qwen3 / GLM-4.5: GQA and MTP support.

### Strong observed gaps

Frontier release notes almost never expose:

- block-table internals,
- native cache surgery APIs,
- rollback semantics,
- capability discovery flags.

That means any Phase C contract here must be a **clean engineering abstraction**,
not a reproduction of public vendor terminology.

## Synthesis from 2025-2026 KV-cache literature

The following primary-source papers were consulted:

1. `Joint Encoding of KV-Cache Blocks for Scalable LLM Serving`
   - arXiv 2601.03067 / OpenReview ICLR 2026
   - explicitly uses the term **KV-cache blocks**
   - focuses on block fusion/shared representations while preserving standard
     cache structure

2. `ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited
   Memory Budget for Long-Context Inference in LLMs`
   - arXiv 2603.08727
   - frames token states as:
     - original/full precision
     - quantized
     - evicted
   - emphasizes adaptive token-importance / per-layer control

3. `KVSink: Understanding and Enhancing the Preservation of Attention Sinks in
   KV Cache Quantization for LLMs`
   - arXiv 2508.04257
   - strongly suggests that sink preservation should be explicit and not just
     "preserve first N tokens"

### What these papers imply for this repo

The literature pushes toward:

- block-oriented cache thinking rather than opaque sequence blobs,
- explicit token-span / block-span accounting,
- adaptive state transitions (retain, quantize, evict),
- explicit sink preservation,
- non-destructive lineage-aware transformations.

It does **not** prescribe a universal operation vocabulary. That part needs to be
chosen here.

## Google / upstream LiteRT-LM fit

The current upstream-compatible shape of this repo strongly suggests Google is
still evolving from a session/checkpoint/replay model, not yet a productionized
native KV surgery interface. The most compatible Phase C contract therefore:

- must degrade cleanly to Phase B replay/recompute,
- must treat snapshots/checkpoints as first-class rollback anchors,
- should add capability discovery explicitly rather than overloading existing
  checkpoint semantics,
- should preserve middleware ownership of memory policy.

This RFC therefore favors a conservative, explicit contract over a highly clever
but implementation-specific interface.

## Proposed Commitments

1. Middleware remains the policy owner.
2. Engine remains the execution owner.
3. Native cache ops are only used when the engine explicitly advertises
   capability support.
4. All destructive native ops are atomic at the operation-group level.
5. On any native-op failure, rollback is attempted first; if rollback cannot be
   proven, the system must fall back to deterministic Phase B recompute.
6. Attention sink preservation is treated as a safety invariant, not an
   optimization.

## KV block model

### Block identity

- `block_id`: immutable identifier for one physical KV allocation unit.
- `session_epoch`: monotonically increasing identifier that changes whenever the
  session is hard-reset or restored from a snapshot lineage root.
- `block_seqno`: monotonically increasing per-session-epoch integer.
- Canonical block identifier format: `(session_epoch, block_seqno)`.

Commitment:

- A `block_id` is never reused within the same `session_epoch`.
- If a block is recreated after compaction or restore, it receives a new
  `block_id` even if the logical tokens are identical.

### Token span

- `token_span`: logical half-open interval `[start_token, end_token)`.
- Spans are expressed in the model's logical prompt timeline, not allocator
  page offsets.
- A block may own:
  - one contiguous token span, or
  - a compacted span set represented as ordered fragments when the engine
    supports block fusion.
- Phase C minimum requirement: contiguous spans.
- Optional extension: fragmented spans for fused/compacted blocks.

Commitment:

- All public range APIs use half-open spans.
- Middleware must never infer physical memory layout from `token_span`.

### Lineage

- `lineage`: append-only ancestry metadata describing how a block came to exist.
- Required lineage fields:
  - `parent_block_ids`
  - `source_op`
  - `source_revision`
  - `created_at_step`
  - `logical_role`
- `source_op` vocabulary:
  - `prefill`
  - `remap`
  - `compact`
  - `snapshot_restore`
  - `delta_apply`

Commitment:

- Lineage forms a DAG.
- A block's lineage is immutable after block creation.
- Compaction produces new blocks with parent references; it does not silently
  mutate lineage in place.

### Required block metadata

Each block must expose:

- `block_id`
- `token_span`
- `lineage`
- `is_pinned`
- `pin_class`
- `heat_score`
- `last_access_step`
- `logical_role`

`pin_class` vocabulary:

- `system_anchor`
- `attention_sink`
- `protected_tail`
- `tool_state`
- `ephemeral`

`logical_role` vocabulary:

- `system`
- `user`
- `assistant`
- `tool`
- `summary_anchor`
- `scratchpad`

## Operation vocabulary

Operations are defined over a `CacheOpGroup`, which is the smallest atomic
commit unit visible to middleware.

### Pin

Purpose:

- Protect selected blocks from eviction, remap invalidation, or compaction
  removal.

Signature:

- `Pin(BlockSelector selector, PinClass pin_class, PinScope scope)`

Behavior:

- Marks matching blocks as pinned.
- `Pin` is idempotent for the same selector and pin class.
- Multiple pin classes may coexist on the same block.

`PinScope` vocabulary:

- `until_unpinned`
- `until_turn_end`
- `until_strategy_transition`

Commitment:

- Blocks pinned as `attention_sink` or `system_anchor` cannot be evicted by any
  best-effort compaction policy.

### EvictRange

Purpose:

- Remove KV state for a logical token interval.

Signature:

- `EvictRange(TokenRange range, EvictMode mode)`

Behavior:

- Evicts blocks fully covered by `range`.
- If a block is only partially covered:
  - minimum-capability engines must reject the op,
  - advanced engines may split the block and return new child blocks.

`EvictMode` vocabulary:

- `strict`
- `best_effort`

Commitment:

- `strict` must fail if pinned or partially covered blocks prevent exact range
  eviction.
- `best_effort` may skip protected blocks, but must report the skipped spans.

### Remap

Purpose:

- Reassign logical token-span ownership without re-prefill.

Signature:

- `Remap(BlockSelector selector, TokenSpan new_span, RemapMode mode)`

Behavior:

- Rebinds selected block metadata to a new logical span.
- Does not change block contents.
- Must preserve lineage by emitting new lineage nodes or explicit remap records.

`RemapMode` vocabulary:

- `logical_only`
- `logical_and_position_adjusted`

Commitment:

- `Remap` must never silently change attention-sink semantics.
- If positional semantics cannot be preserved, the op must fail.

### Compact

Purpose:

- Reduce memory footprint by replacing one or more source blocks with a smaller
  target representation.

Signature:

- `Compact(CompactPlan plan)`

Required `CompactPlan` fields:

- `source_blocks`
- `target_budget_tokens`
- `retained_selectors`
- `summary_anchor_policy`
- `preserve_pins`

Behavior:

- Produces a new block set.
- Invalidates or tombstones source blocks only after commit.
- May emit a `summary_anchor` block when summary-assisted compaction is used.

Commitment:

- `Compact` is never in-place at the metadata level; it produces a new lineage
  boundary.
- If summary generation is required and the summary artifact is unavailable, the
  op must fail before commit.

### SnapshotRestore

Purpose:

- Capture and restore a full cache state boundary.

Signature:

- `SnapshotRestore::CreateSnapshot(SnapshotLabel label)`
- `SnapshotRestore::Restore(SnapshotLabel label)`

Behavior:

- Snapshot captures:
  - block table
  - pin metadata
  - lineage heads
  - allocator generation
  - current logical step
- Restore atomically replaces current visible cache state with the snapshot.

Commitment:

- Restore creates a new `session_epoch`.
- Restored blocks retain provenance via lineage references to the snapshot root.

## Failure semantics

### Atomicity

- All native cache ops are executed inside a `CacheOpGroup`.
- Visibility rule: either the entire group commits, or none of it becomes
  visible to middleware.
- Intermediate allocator state must never be visible outside the op group.

Commitment:

- Middleware may assume read-after-commit consistency at op-group boundaries.

### Rollback guarantees

- Every op group must define:
  - pre-state snapshot handle,
  - touched block set,
  - rollback viability flag.
- If commit fails before visibility, engine must roll back internally.
- If commit fails after partial internal mutation and rollback viability is
  false or unproven, engine must report `rollback_unavailable`.

Commitment:

- The engine must never return success for an op group with unresolved partial
  mutation.

### Failure result vocabulary

Required failure codes:

- `unsupported_capability`
- `invalid_selector`
- `range_conflict`
- `pinned_block_conflict`
- `position_semantics_violation`
- `summary_artifact_missing`
- `snapshot_not_found`
- `rollback_unavailable`
- `internal_cache_corruption_suspected`

### Fallback to Phase B recompute

Fallback rule:

- On `unsupported_capability`, `rollback_unavailable`, or
  `internal_cache_corruption_suspected`, middleware must abandon the native path
  for the current transition and invoke deterministic Phase B recompute.

Commitment:

- Phase B recompute is the universal escape hatch.
- Native failure must not strand the conversation in a partially shifted state.

## Capability discovery vocabulary

Capability discovery is explicit and versioned.

### Required capability fields

- `supports_kv_surgery`
- `supports_attention_sink_pinning`
- `supports_range_evict`
- `supports_block_remap`

### Recommended capability fields

- `supports_compact`
- `supports_snapshot_restore`
- `supports_partial_block_split`
- `supports_summary_anchor_blocks`
- `supports_position_adjusted_remap`

### Capability rules

- `supports_kv_surgery=false` means middleware must stay on the Phase B path.
- `supports_range_evict=true` does not imply `supports_block_remap=true`.
- `supports_attention_sink_pinning=true` is required before any native
  compaction strategy that would otherwise touch sink-adjacent blocks.
- Capability values are immutable for the life of a session.

## Comparison back to this repo

### Current repo state

Current middleware/runtime in `Conversation` already has:

- replay-pack identity
- retained slice metadata
- validity hashes
- policy digests
- checkpoint-based fallback / replay

But it does **not** yet have:

- builder identity
- native cache capability discovery
- block-level metadata
- native cache-op vocabulary

### Required Phase B alignment changes implied by this RFC

To keep Phase B compatible with this RFC:

1. Middleware prefetch artifacts should grow explicit identity for:
   - builder id
   - retained ranges
   - protected ranges
   - summary anchor presence
2. Phase B telemetry should emit reason codes that can later be shared by native
   and middleware paths.
3. Policy selection should remain middleware-owned even after native ops exist.

## Non-goals

- Defining allocator page size.
- Mandating a vendor-specific cache table layout.
- Claiming frontier labs publicly expose these exact native cache APIs.
- Replacing Phase B fallback.
