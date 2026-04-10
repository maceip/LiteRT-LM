# Phase C Bootstrap: Native Cache Operations with Phase A Look-Back

This document defines how to begin **Phase C** after Phase B middleware work
has been completed and signed off.

Phase C introduces **engine-native cache operations** behind explicit
capability discovery while preserving:

- middleware ownership of policy,
- engine ownership of execution,
- deterministic Phase B recompute as the universal escape hatch.

The Phase C contract is defined by
`docs/PHASE_C_CACHE_OPS_RFC_DRAFT.md`. This bootstrap document describes the
practical startup sequence and the validation steps which must happen before
native work begins.

## Preconditions

Phase C should not start until all of the following are true:

- Phase A safety/gating is complete.
- Phase B middleware gating is complete.
- The Phase C RFC vocabulary is accepted as the source of truth for native
  operation names, failure codes, and capability discovery.

Required reference documents:

- `docs/PHASE_A_GATE.md`
- `docs/PHASE_B_GATE.md`
- `docs/PHASE_C_CACHE_OPS_RFC_DRAFT.md`

## Step 0: Phase A look-back validation

Before any engine-native work is started, perform a **Phase A look-back
validation** to ensure Phase C is built on top of the same safety guarantees
that Phase B depends on.

The Phase A look-back must confirm:

1. **Safe-boundary queueing still holds**
   - policy changes only apply at:
     - `tool_result`
     - `turn_boundary`

2. **Atomic-turn enforcement still holds**
   - no runtime policy transition during active prefill/decode
   - no runtime policy transition during append mode

3. **Priority arbiter behavior is still intact**
   - profile hard constraints first
   - runtime overrides second
   - model/runtime hard limits last

4. **Transition-note behavior is still correct**
   - transition markers remain boundary-safe
   - no role/channel regressions were introduced by Phase B

5. **Version / compatibility gating still rejects safely**
   - unsupported values fail before mutating active runtime state

Why this matters:

Phase C native operations must never become a side channel that bypasses the
control-plane guarantees already established in Phase A.

## Step 1: Phase B look-back validation

Before beginning native implementation, confirm the final Phase B bridge state
is present and correct.

Required middleware state:

- async background prefetch planning
- safe-boundary-only install
- builder identity
- retained ranges
- protected ranges
- summary-anchor presence
- structured reason-attributable telemetry
- deterministic fallback to Phase B recompute

Why this matters:

Phase C should extend the middleware contract, not replace it. Native work must
plug into the same policy/fallback/telemetry framework established by Phase B.

## Step 2: Lock the ownership model

Phase C must preserve the ownership split already established in the RFC:

- **middleware owns policy**
- **engine owns execution**
- **native cache ops are used only when capability support is explicitly
  advertised**

Implications:

- middleware chooses strategy / fallback
- engine performs `CacheOpGroup` execution
- unsupported or unsafe native states must fall back to deterministic Phase B
  recompute

## Step 3: Add capability discovery first

Do not start by implementing individual cache verbs.

Start with capability discovery so middleware can decide whether the native
path is even legal for a given session.

Required capability fields:

- `supports_kv_surgery`
- `supports_attention_sink_pinning`
- `supports_range_evict`
- `supports_block_remap`

Recommended capability fields:

- `supports_compact`
- `supports_snapshot_restore`
- `supports_partial_block_split`
- `supports_summary_anchor_blocks`
- `supports_position_adjusted_remap`

Rules to enforce immediately:

- `supports_kv_surgery=false` means remain on Phase B
- capability values are immutable for the lifetime of a session
- capability support must not be inferred transitively

## Step 4: Define the native execution envelope

Before implementing individual verbs, define the execution envelope around
them.

That means:

- `CacheOpGroup` atomicity
- rollback bookkeeping
- failure result vocabulary
- middleware fallback hooks

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

Fallback rule to preserve:

On:

- `unsupported_capability`
- `rollback_unavailable`
- `internal_cache_corruption_suspected`

middleware must abandon the native path and run deterministic Phase B
recompute.

## Step 5: Implement metadata before mutators

The first engine-native step should expose or internally define the block model
required by the RFC, before attempting destructive operations.

Required metadata concepts:

- `block_id = (session_epoch, block_seqno)`
- contiguous `token_span`
- immutable lineage metadata
- pin classes
- logical roles

Why this matters:

Without explicit metadata, native mutators risk becoming allocator-specific,
opaque, and impossible to validate against middleware policy.

## Step 6: Introduce verbs incrementally

Native verbs should be staged in the same order that minimizes correctness
risk.

Recommended order:

1. `Pin`
2. `EvictRange`
3. `Remap`
4. `Compact`
5. `SnapshotRestore`

Do not expose a verb to middleware until:

- capability discovery exists,
- op-group atomicity exists,
- rollback semantics are defined,
- failure vocabulary is wired,
- deterministic Phase B fallback is proven.

## Step 7: Keep scaffold honesty

If a native summary-anchor or quarantine-adjacent capability is not really
implemented yet, Phase C bootstrap must not pretend otherwise.

Explicitly distinguish:

- implemented native operation
- capability stub
- middleware scaffold
- unsupported path requiring fallback

This mirrors the same honesty rule already applied to the Phase B scaffold
builders.

## Step 8: Conformance tests required during bootstrap

Phase C bootstrap should not be considered complete without at least these test
buckets:

1. **Phase A look-back validation**
   - boundary safety still enforced with native path disabled/enabled

2. **Capability-gating tests**
   - unsupported sessions remain on Phase B path

3. **Atomicity / rollback tests**
   - partial native mutation cannot surface as success

4. **Fallback tests**
   - required failure codes trigger deterministic Phase B recompute

5. **Vocabulary / metadata conformance tests**
   - block identity, pin class, and operation naming align with the RFC

## Non-goals for bootstrap

Phase C bootstrap does **not** require:

- immediate implementation of every native verb,
- optimizer-level compaction sophistication,
- vendor-specific allocator semantics,
- removal of the Phase B path.

## Bootstrap exit criteria

Phase C bootstrap is complete only when:

- Phase A look-back validation is complete and documented
- capability discovery exists and is immutable for session lifetime
- `CacheOpGroup` execution + rollback envelope exists
- required failure vocabulary is wired to middleware fallback
- deterministic Phase B fallback remains intact
- the first native path can be enabled only through explicit capabilities

At that point, Phase C implementation can proceed incrementally without
weakening the Phase A/Phase B guarantees already in place.
