# Phase A Gate: Hybrid Runtime Memory Policy Safety

This document defines the required completion criteria for **Phase A** of the
hybrid memory-policy rollout.

Phase A scope is control-plane safety and deterministic policy transitions.
It does **not** require engine-native KV remapping.

## Objectives

1. Runtime policy changes are safe and deterministic.
2. Policy changes never apply mid-turn or mid-tool-use chain.
3. YAML/profile constraints are enforced before any runtime override.
4. Behavior is observable and testable.

## In Scope

- Runtime policy parsing and validation.
- Safe-boundary policy queuing and application.
- Atomic-turn enforcement for sync and async paths.
- Priority arbitration.
- Transition-note support.
- Unit tests for all of the above.

## Out of Scope

- Predictive prefetching and precomputed replay packs (Phase B).
- KV-cache range remap / tensor-level cache surgery (Phase C).
- Attention sink pinning in executor attention masks (Phase C).

## Required Functional Behavior

### 1) Safe-boundary queueing

- If a policy update is requested while a turn is active, runtime must queue it.
- Queued policy must apply at configured boundary only:
  - `tool_result`
  - `turn_boundary`

### 2) Atomic-turn rule

- Runtime must not apply a new policy while prefill/decode is in progress.
- Runtime must not apply a new policy during appending mode.

### 3) Priority arbiter

Enforcement order:

1. Profile hard constraints (`allow_runtime_tuning`, schema/version gates).
2. Runtime override fields.
3. Model/runtime hard limits (token budget, unsupported session capabilities).

### 4) Transition note

- If `emit_transition_note=true`, add an internal transition marker at boundary.
- Transition marker must not violate role alternation or channel-content logic.

### 5) Version/compatibility gating

- Unsupported `version` / `compatibility` must fail early.
- Failures must not mutate active policy state.

## Test Matrix (Phase A)

Minimum required tests:

1. Parse profile with all supported strategy ids.
2. Parse and enforce new control-plane fields:
   - `version`
   - `compatibility`
   - `allow_runtime_tuning`
   - `safe_boundary`
   - `shadow_strategy`
   - `emit_transition_note`
3. Queue update then apply at boundary (sync path).
4. Queue update then apply at boundary (async path).
5. Reject override when `allow_runtime_tuning=false`.
6. Reject invalid profile compatibility/version.
7. No regression in existing context-shift tests.

## Gate Checklist

Phase A is complete only when all items are true:

- [ ] Safe-boundary queue methods are implemented and wired.
- [ ] Sync and async message paths both enforce atomic-turn transitions.
- [ ] Priority arbiter logic is enforced in runtime.
- [ ] Transition-note behavior is implemented and tested.
- [ ] Version/compatibility reject path is implemented and tested.
- [ ] Existing context-shift tests still pass.
- [ ] New tests for boundary queueing and constraints pass.
- [ ] Changes are committed, pushed, and PR notes updated.

## Suggested Evidence for Sign-off

- Test names and pass results.
- Before/after policy timeline for one sync and one async run.
- Example failed override showing expected rejection status.
- Example transition-note insertion at boundary.
