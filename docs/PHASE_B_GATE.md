# Phase B Gate: Async Prefetch Middleware Completion

This checklist defines what must be true before **Phase B** can be declared
complete.

Phase B remains a **middleware-owned** replay/install system. It does **not**
introduce Phase C engine-native KV surgery. Deterministic Phase B recompute
remains the universal escape hatch.

## Required functional state

- [x] Prefetch planning is no longer performed inline from
      `MaybeApplyContextShift()`.
- [x] Prefetch planning runs through a dedicated background queue.
- [x] Install still happens only at safe boundary.
- [x] Planner lifecycle exposes:
  - `Planned`
  - `Computing`
  - `Ready`
  - `Installed`
  - `Discarded`
- [x] Superseding-plan invalidation works even when an older task is already
      running, via plan-token / publish-if-current guards.
- [x] Queued job cancellation exists for:
  - queued/applied policy change
  - history revision mismatch
  - superseding plan

## Required RFC bridge state

Phase B artifacts must now carry explicit middleware identity that aligns with
the RFC bridge section:

- [x] `builder_id`
- [x] retained ranges
- [x] protected ranges
- [x] summary-anchor presence

Current builder ids:

- `replay_recent`
- `drop_all_but_system`
- `summarize_protected_tail`
- `quarantine_merge`

## Scaffold honesty requirements

`summarize_protected_tail` and `quarantine_merge` are currently **Phase B
scaffolds**, not full semantic transforms.

That means they must remain:

- deterministic
- explicitly identified in artifacts and telemetry
- explicit about scaffold parity/fallback semantics
- documented as not yet creating a real summary artifact or quarantine store

The current implementation satisfies this by:

- emitting explicit builder identity
- emitting retained/protected ranges
- marking scaffold builders with semantic-parity-oriented metadata
- keeping deterministic replay install / fallback behavior

## Telemetry requirements

Prefetch metrics must expose dimensions that can later align with native Phase
C failure accounting.

Required dimensions:

- [x] `profile_id`
- [x] strategy
- [x] builder id
- [x] boundary
- [x] model type
- [x] reason code

Required outcome/event coverage:

- [x] planned
- [x] installed
- [x] stale discarded
- [x] shadow skipped
- [x] install failed
- [x] fallback

Parity requirements:

- [x] parity mode is explicit
- [x] strict-token parity is available where applicable
- [x] semantic parity / scaffold-oriented mode is explicit where applicable
- [x] runtime correctness does not depend on clone support

## Test evidence required

Phase B is not complete until all four evidence buckets exist:

### 1) Unit coverage

- [x] install-hit path
- [x] stale retained-range/history mismatch discard
- [x] runtime-policy-change discard
- [x] useful-plan reuse
- [x] builder/telemetry identity assertions

### 2) Concurrency coverage

- [x] async post-boundary planning path
- [x] superseding queued plan invalidation
- [x] publish-if-current / stale-job suppression coverage

### 3) Long-session integration coverage

- [ ] repeated multi-turn near-threshold session with multiple queued plans,
      installs, and deterministic fallbacks

### 4) Performance comparison evidence

- [ ] controlled comparison showing install-hit path improves over baseline
      recompute path using recorded latency totals

## Commands / evidence currently used

Focused automated evidence already exercised during execution:

- `bazel test //runtime/framework:execution_queue_test`
- `bazel test //runtime/conversation:conversation_test --test_filter='ConversationTest.PrefetchReplayPackInstallsOnBoundaryWhenValid|ConversationTest.PrefetchInstallDiscardedOnRetainedSliceDigestMismatch|ConversationTest.PrefetchPlanDiscardedWhenRuntimePolicyChanges|ConversationTest.PrefetchPlannerRunsAsynchronouslyAfterBoundary|ConversationTest.SupersedingQueuedPlanRemovesOlderPendingTask|ConversationTest.PrefetchMetricsCaptureStructuredDimensions'`

## Remaining gate work

Phase B should **not** be declared complete until the unchecked items below are
finished:

- [ ] long-session integration evidence
- [ ] explicit performance comparison evidence

Everything else in this checklist is now in place on the middleware side.
