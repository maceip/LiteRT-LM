# Context Memory Catalog

This catalog defines versioned, reusable runtime memory profiles for
`ConversationConfig::RuntimeMemoryPolicy`.

## Current Implementation Matrix

This reflects where the current LiteRT-LM implementation is today.

| Capability area | Status | Notes |
| --- | --- | --- |
| Phase A control-plane safety | Implemented | Safe-boundary application, policy queueing, atomic-turn protections. |
| Phase B predictive prefetch | Implemented (middleware) | Async planning/install with validity checks and deterministic fallback. |
| Phase C native cache ops | Bootstrap | Capability-gated native path with fallback to Phase B recompute. |
| KV cache remap/compact surgery | Partial | Bootstrap includes native evict flow; full remap/compact semantics are pending. |
| Attention sink preservation | Partial | Protected head behavior exists; true engine-level sink pinning/mask controls are pending. |
| Policy-aware/token-metadata policy surface | Partial | YAML parser now preserves metadata annotations and parses structured v2 policy blocks; token-level pruning logic is still pending. |
| Traceable tensor/attention diagnostics | Not implemented | No persisted attention heatmap or token-attention tracing yet. |

## Catalog Profiles

### Frozen baseline

| Catalog entry | File | Intent |
| --- | --- | --- |
| `catalog.step_frozen.v1` | `runtime/conversation/testdata/context_catalog_step_frozen.yaml` | Freeze the Step-style discard-all reset behavior. |

### Zugzug profile

| Catalog entry | File | Intent |
| --- | --- | --- |
| `zugzug.context_manager.v1` | `runtime/conversation/testdata/zugzug_catalog.yaml` | Production-oriented variant preserving reset semantics with orchestration metadata hints. |

### Full 16-strategy best-in-class profile set

All strategy profiles live under:

- `runtime/conversation/testdata/catalog_profiles/`

Each file includes:

- profile metadata (author/source repo/source notes)
- executable runtime policy fields
- compact `catalog_meta` strengths/failure-modes

## v2 expressive policy blocks

Catalog profiles may define the following structured blocks:

- `execution.native_cache_ops.*`
- `execution.protected_ranges.*`
- `execution.fallback_policy.*`
- `prefetch.planner.*`
- `verification.*`
- `telemetry.*`
- `orchestration.external_memory.*`

`ParseMemoryPolicyYaml` maps these into `RuntimeMemoryPolicy` fields, and
retains additional dotted keys in `RuntimeMemoryPolicy.metadata` for tooling.

## Strategy-to-file map

1. `hard_reset_replay_window`:
   `01_hard_reset_replay_window.yaml`
2. `summarize_protected_tail`:
   `02_summarize_protected_tail.yaml`
3. `virtual_memory_paging`:
   `03_virtual_memory_paging.yaml`
4. `fact_memory_extraction_update`:
   `04_fact_memory_extraction_update.yaml`
5. `semantic_compression_consolidation_adaptive_retrieval`:
   `05_semantic_compression_consolidation_adaptive_retrieval.yaml`
6. `learned_compression_policy`:
   `06_learned_compression_policy.yaml`
7. `incremental_hierarchical_aggregation`:
   `07_incremental_hierarchical_aggregation.yaml`
8. `active_recall_surprise_update`:
   `08_active_recall_surprise_update.yaml`
9. `contextual_forgetting_interference_management`:
   `09_contextual_forgetting_interference_management.yaml`
10. `token_efficient_kv_cache_management`:
    `10_token_efficient_kv_cache_management.yaml`
11. `reflection_metacognitive_buffering`:
    `11_reflection_metacognitive_buffering.yaml`
12. `self_correcting_fact_graph`:
    `12_self_correcting_fact_graph.yaml`
13. `slow_fast_memory_architecture`:
    `13_slow_fast_memory_architecture.yaml`
14. `heat_based_tiered_migration`:
    `14_heat_based_tiered_migration.yaml`
15. `context_quarantine_isolated_scratchpads`:
    `15_context_quarantine_isolated_scratchpads.yaml`
16. `mcp_active_metadata`:
    `16_mcp_active_metadata.yaml`
