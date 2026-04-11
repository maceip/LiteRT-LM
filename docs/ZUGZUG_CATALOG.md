# Zugzug Catalog

The Zugzug catalog tracks context-memory variants that extend the frozen
Step-style reset baseline for production orchestration.

## Current entry

- Profile id: `zugzug.context_manager.v1`
- YAML: `runtime/conversation/testdata/zugzug_catalog.yaml`

## Design intent

Keep deterministic context-shift controls in runtime while enabling a higher
layer (agent/gateway) to persist and rehydrate curated memory across reset
episodes.

The runtime-enforced policy remains strict and bounded. Extra keys under
`external_memory.*` are annotations for orchestration code and are currently
ignored by `ConversationConfig::ParseMemoryPolicyYaml`.

## Engine-native implementation notes (detail)

The following patterns capture the intended production direction and map to the
current phased roadmap:

1. **KV-cache surgical editing**
   - Triggered by profile policy (for example, summarize/protected-tail modes).
   - Engine-native objective: re-map or evict cache ranges while preserving
     protected head/tail spans, avoiding full prompt re-tokenization.
2. **Attention sink preservation**
   - Preserve first-anchor tokens under all active policies.
   - Prevent instability/perplexity spikes after context pruning.
3. **Predictive context prefetch**
   - Plan compression/replay packs asynchronously ahead of boundary events.
   - Install only at safe boundaries with strict staleness and policy checks.
4. **Policy-aware token handling**
   - Catalog can carry metadata intent for selective handling
     (facts/code/chitchat/etc.).
   - Runtime parser currently enforces only declared `RuntimeMemoryPolicy`
     fields; metadata extensions remain orchestration-layer hints.
5. **Traceable execution diagnostics**
   - Future target: persist attention/selection diagnostics for failure
     forensics and parity checks.

## Sidecar vs engine-native (current comparison)

| Feature | Sidecar (middleware) | Engine-native target |
| --- | --- | --- |
| Granularity | string/token replay planning | tensor/KV block operations |
| Latency profile | recompute cost on fallback | near-zero on successful native ops |
| State consistency | can require replay reconciliation | direct cache-state transition |
| Compute overhead | pays to re-read retained context | pays mainly for new tokens |

## Status

- Phase A: control-plane safety in place.
- Phase B: predictive prefetch in place (middleware).
- Phase C: native cache-op bootstrap present; full remap/compact/pinning stack
  remains in-progress.
