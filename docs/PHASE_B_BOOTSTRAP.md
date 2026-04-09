# Phase B Bootstrap: Predictive Prefetch and Precomputed Replay Packs

This document defines how to start **Phase B** after Phase A gate is complete.

Phase B adds runtime performance improvements without requiring engine-native
KV-cache remapping.

## Goals

1. Reduce next-turn latency near context-shift thresholds.
2. Keep correctness identical to baseline (non-prefetch) behavior.
3. Ship behind a feature flag with safe fallback.

## Dependencies

Phase B starts only after Phase A is green:

- Safe-boundary transitions are implemented and tested.
- Policy arbitration is deterministic.
- Transition-note and constraints are stable.

## Core Concepts

### Prefetch Planner

When context usage approaches trigger ratio, runtime starts background planning:

- selects candidate retention window (based on active strategy),
- renders replay text or prepares `InputData`,
- records source checkpoint and history watermark.

### Precomputed Replay Pack

A replay pack is a prebuilt artifact used at boundary-time:

- active `profile_id` and strategy snapshot,
- source checkpoint label/hash,
- retained message slice metadata,
- replay payload (`InputData` or rendered text),
- target ratio and retain count used,
- validity fields (history size, last message index, session step).

### Boundary Installer

At safe boundary:

1. validate pack freshness,
2. if valid, apply pack,
3. if stale/invalid, fall back to existing synchronous replay path.

## Implementation Steps

### B1) Add feature flags

- `prefetch_enabled` (default false)
- `prefetch_shadow_mode` (compute only, no install)
- `prefetch_min_ratio` (start planning threshold, e.g. 0.75)

### B2) Define replay-pack structures

Add a runtime-only struct near conversation policy internals:

- `PrecomputedReplayPack`
- `PrefetchPlannerState`

### B3) Trigger planner

Planner trigger location:

- after decode completion and before returning next turn, when
  `current_step / max_tokens >= prefetch_min_ratio`

### B4) Install at safe boundary

Use existing Phase A boundary hooks:

- on `tool_result` or `turn_boundary`, attempt install if pack matches policy.

### B5) Fallback correctness path

- Any mismatch, stale watermark, or session capability mismatch => fallback.
- Never block user turn on planner thread completion.

## Metrics (Required)

Capture at minimum:

- prefetch hit rate,
- install success rate,
- stale/discard rate,
- next-turn latency delta vs baseline,
- token/compute overhead of planner,
- correctness parity (response/task regression rate).

## Rollout Plan

1. **Shadow mode**: build packs, never install.
2. **Canary install**: low traffic / test profiles only.
3. **Wider rollout** with automatic rollback on error thresholds.

## Success Criteria (Phase B Gate)

- [ ] Prefetch planner is non-blocking and bounded in resource usage.
- [ ] Replay-pack install works only at safe boundaries.
- [ ] Correctness parity with baseline is demonstrated.
- [ ] Median and p95 next-turn latency improve for near-threshold turns.
- [ ] Fallback path is exercised and verified.
- [ ] Metrics dashboard covers hit/miss/install/error outcomes.

## Notes for Phase C Hand-off

Phase B artifacts should be designed so they can later map to Phase C
engine-native KV edit plans:

- keep pack metadata explicit (head/tail ranges),
- preserve policy snapshot IDs,
- maintain deterministic boundary logs.
