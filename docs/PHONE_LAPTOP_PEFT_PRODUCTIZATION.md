# Phone Harness + Laptop GPU PEFT Productization Plan

This document defines how to productize the primary lane:

- phone harness as the user-facing runtime and data/eval capture point,
- laptop-side fine-tuning via PEFT (LoRA/QLoRA),
- overnight job execution with quality gates.

It is written to move from investigation to implementation.

## 1) Problem Statement

Users want personalization and task adaptation without cloud-only lock-in.
Pure on-device training for larger models is inconsistent due to thermal,
power, and throughput constraints. We need a path that is practical,
private-by-default, and shippable.

## 2) Product Decision

Default product lane:

1. Phone runs inference + capture/eval harness.
2. Laptop runs a sidecar training service using GPU PEFT (LoRA/QLoRA).
3. Training runs overnight with strict promotion gates.
4. Optional cloud burst is a fallback when laptop capacity is insufficient.

## 3) Design Principles

1. **User control first**: explicit opt-in, transparent data and model use.
2. **Private by default**: local network flow, encrypted artifacts at rest.
3. **Safe promotion**: no model swap without evaluation pass.
4. **Deterministic operations**: resumable jobs, idempotent state transitions.
5. **Operational simplicity**: one local sidecar process and clear health checks.

## 4) System Architecture

## 4.1 Components

### A) Phone Harness App

- Runs baseline/tuned model inference.
- Captures user-approved training examples and eval suites.
- Schedules and requests overnight jobs.
- Receives candidate adapters and evaluation reports.

### B) Adapter Shim (SDK + Protocol)

The Adapter Shim is the compatibility layer between phone runtime formats and
training artifacts.

Responsibilities:

- Normalize datasets from harness format into trainer format.
- Normalize adapters/checkpoints back into runtime-loadable format.
- Maintain versioned schema compatibility between app, sidecar, and model pack.

### C) Laptop Sidecar Service

Local service installed with desktop companion.

Responsibilities:

- Watches job queue from phone.
- Runs PEFT training/evaluation pipelines on laptop GPU.
- Persists artifacts and metrics.
- Exposes local APIs for status, logs, and candidate promotion.

### D) Optional Cloud Burst Broker

- Triggered only when policy detects insufficient local resources.
- Reuses same job spec and promotion contract as sidecar.

### E) Artifact Store (Local First)

- Base model references.
- Adapter outputs (`adapter.safetensors`, metadata, eval report).
- Merged export if needed by serving target.

## 4.2 Logical Data Flow

1. User opts in to personalization and nightly training.
2. Phone harness collects approved examples and eval fixtures.
3. Adapter Shim serializes a `TrainingJobSpec`.
4. Sidecar admits job, validates resources, and starts run.
5. Sidecar trains adapter (LoRA/QLoRA), then evaluates candidate.
6. Candidate + eval report returned to phone.
7. Promotion gate checks quality/safety thresholds.
8. If pass, phone activates candidate adapter at next safe boundary.

## 5) Adapter Shim Contract (v0)

The shim must be versioned and backward-compatible within major version.

## 5.1 `TrainingJobSpec`

Required fields:

- `job_id`
- `schema_version`
- `model_id` (base)
- `target_variant` (`E2B`, `E4B`, etc.)
- `trainer_mode` (`lora`, `qlora`)
- `dataset_ref`
- `eval_suite_ref`
- `policy` (budget + promotion gates)

## 5.2 `TrainingJobResult`

Required fields:

- `job_id`
- `status` (`succeeded`, `failed`, `aborted`)
- `adapter_ref`
- `metrics` (loss, eval summary, runtime stats)
- `safety_report`
- `compatibility` (runtime + schema)

## 5.3 Compatibility Rules

1. Reject unknown major schema.
2. Allow minor additive fields.
3. Reject adapter if runtime capability mismatch.
4. Never partially activate candidate artifacts.

## 6) Overnight Scheduler Policy

## 6.1 Admission

Accept job only if:

- device is charging,
- user is idle within configured window,
- laptop sidecar is healthy and reachable.

## 6.2 Budgeting

Per run policy includes:

- max wall runtime,
- max expected steps,
- minimum checkpoint cadence,
- cancellation conditions (thermal/resource/safety).

## 6.3 Completion Policy

If budget exhausted:

- keep latest checkpoint,
- run lightweight eval,
- return `partial_success` with recommendation (`continue` or `stop`).

## 7) Model Strategy

1. **Fast lane**: start adaptation on smaller variant for rapid iteration.
2. **Promotion lane**: run larger variant only when fast-lane eval gains persist.
3. **Fallback**: if large-model job misses quality/compute gates, keep current
   production adapter.

## 8) Quality Gates

A candidate is promotable only when all pass:

1. Task success rate improvement against held-out eval suite.
2. No regression on safety and refusal checks.
3. Output style/format conformance within policy bounds.
4. Runtime compatibility validation passes.

No promotion based only on training loss.

## 9) Security and Privacy Requirements

1. Explicit consent for data capture and training.
2. Per-example user review controls and deletion support.
3. Local-network TLS between phone app and sidecar.
4. Artifact encryption at rest with rotating keys.
5. Auditable job/activity log with redaction for sensitive content.

## 10) Reliability and Observability

## 10.1 Sidecar Health

- `ready`, `degraded`, `unavailable` states.
- GPU capability probe on startup.
- queue depth, job latency, fail rate, checkpoint freshness.

## 10.2 Core Metrics

- job admission rate
- completion rate
- promotion pass rate
- regression rejection rate
- median and p95 training-to-promotion latency

## 10.3 Failure Handling

- resumable checkpoints
- idempotent job retries
- deterministic rollback to last known good adapter

## 11) Product Backlog (Execution-Ready)

## Epic A: Adapter Shim v0

- Define `TrainingJobSpec` and `TrainingJobResult` schemas.
- Build serializers/deserializers for phone and sidecar.
- Add compatibility validator and contract tests.

Exit criteria:

- end-to-end schema test passing for success/failure/partial flows.

## Epic B: Laptop Sidecar MVP

- Local daemon with queue, lifecycle, health API.
- GPU capability detection and trainer worker orchestration.
- Artifact persistence and signed manifest generation.

Exit criteria:

- sidecar runs one complete train-eval cycle from phone-submitted job.

## Epic C: Overnight Scheduler + Policies

- Charging/idle admission rules.
- Runtime budget and checkpoint controls.
- Cancel/resume behavior with state persistence.

Exit criteria:

- deterministic overnight run behavior across three restart scenarios.

## Epic D: Promotion Gate + Runtime Activation

- Gate engine consuming eval/safety reports.
- Safe adapter activation boundary in phone runtime.
- Rollback API and UI control.

Exit criteria:

- candidate promotion/rollback proven with integration tests.

## Epic E: Trust and Ops

- Consent UX and data controls.
- Audit logs and metrics dashboard.
- Incident playbook for failed/unsafe candidates.

Exit criteria:

- security review checklist complete and observability baseline live.

## 12) Delivery Sequence (No Calendar Estimates)

1. Build and freeze Adapter Shim v0 contract.
2. Ship sidecar MVP with single-model LoRA path.
3. Integrate overnight scheduler and checkpoint resume.
4. Add promotion gates and rollback.
5. Expand to QLoRA and optional cloud burst routing.

## 13) Risks and Mitigations

1. **Resource variability on laptops**
   - Mitigation: strict capability probe + dynamic policy + cloud burst fallback.
2. **Adapter/runtime incompatibility**
   - Mitigation: shim compatibility validator + activation-time checks.
3. **Silent quality regressions**
   - Mitigation: hard promotion gates and held-out eval suites.
4. **User trust/privacy concerns**
   - Mitigation: transparent controls, local-first defaults, full auditability.

## 14) Immediate Next Actions

1. Finalize shim schema and sign-off owners.
2. Stand up sidecar prototype with one training backend.
3. Implement minimal promotion gate with pass/fail report contract.
4. Run pilot with internal datasets and iterate on gate thresholds.
