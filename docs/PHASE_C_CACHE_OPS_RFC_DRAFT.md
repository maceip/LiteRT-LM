# Phase C Cache Operations RFC Draft

This document defines the middleware/engine contract for Phase C engine-native
context management.

It is intentionally conservative and is designed to evolve from the current
Phase B black-box session model, which today exposes checkpoint, rewind, clone,
prefill, decode, and current-step operations.

## Goals

1. Make KV cache state addressable without requiring full replay/prefill.
2. Provide a small, explicit vocabulary for cache surgery.
3. Guarantee deterministic failure handling and fallback to Phase B recompute.
4. Keep the contract implementable on top of the current engine/session model.

## Commitments

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
- Spans are expressed in the model's logical prompt timeline, not in allocator
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

## Phase B / Phase C bridge commitments

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
- Replacing Phase B fallback.
