# GitHub Issue (posted upstream)

**Repository:** google-ai-edge/LiteRT-LM
**Title:** `[Feature Request] Context window management for long-running conversations`

---

LiteRT-LM ships [`SaveCheckpoint`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h#L218), [`RewindToCheckpoint`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h#L223), [`GetCurrentStep`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h#L228), [`ClearKVCache`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/executor/llm_litert_npu_compiled_model_executor.h#L327), and [`DeleteTokensFromKvCache`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/executor/llm_litert_compiled_model_cache_utils.h#L45) — but no coordinating layer that uses them together when a conversation approaches `max_num_tokens`. On a 4K-context edge device, that wall gets hit fast, especially with agentic tool-calling workflows.

The primitives clearly anticipate this. Is there an internal plan that hasn't been published yet?

We've prototyped a solution in [this branch](https://github.com/maceip/LiteRT-LM/tree/cursor/-bc-fc50080b-70ce-47a5-b93d-931eaf8184e6-8ef1) — a session-level context shift system on [`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h#L250) with configurable trigger/target ratios, two eviction strategies (`kReplayRecent` with iterative budget-fitting, `kDropAllButSystem`), and a reset-on-exhaustion fallback. It's checkpoint-based rather than ring-buffer-based to preserve KV cache coherence, and auto-disables on backends that don't support checkpointing.

Would love to know:
1. Is context lifecycle management planned for `Conversation`, or is it intended to stay application-layer?
2. Are contributions welcome for something this architectural?

Happy to open a PR if there's interest.
