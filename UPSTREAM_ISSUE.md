# GitHub Issue: Request for Clarity on Context Window / Memory Management Strategy for Long-Running Conversations

**Repository:** google-ai-edge/LiteRT-LM
**Title:** `[Feature Request] Document and formalize context window management strategy for long-running conversations`

---

## Issue Body

### Summary

LiteRT-LM has invested significantly in KV cache infrastructure (#1807 DeepCopy, #1680 cache clearing, #1800 checkpoint-based filtering, #1601/#1491 LiteRT-based KV cache + interfaces), yet there is **no documented strategy or public API** for what happens when a long-running conversation approaches or exceeds `max_num_tokens`. This is a critical gap for any production deployment of on-device LLMs in conversational settings — and the building blocks to solve it already exist in the framework.

### Problem

When deploying LLMs on edge devices with constrained context windows (e.g., Gemma3-1B at 4096 tokens), real-world multi-turn conversations will inevitably exceed the context limit. Today, LiteRT-LM provides:

- `Session::SaveCheckpoint()` / `Session::RewindToCheckpoint()` — named checkpoint support
- `Session::GetCurrentStep()` — current token position querying
- `ClearKVCache()` — full KV cache reset
- `DeleteTokensFromKvCache()` — ring-buffer-style token rotation (StreamingLLM pattern)
- `Engine::CreateSession()` — session lifecycle management

However, there is **no coordinating layer** that uses these primitives to automatically manage context pressure during conversations. Developers integrating LiteRT-LM must currently build this themselves, with no guidance on the intended approach.

### Questions for the Team

1. **Is there an internal strategy** for session-level context window management that hasn't been published yet? The checkpoint and KV cache primitives suggest this was anticipated.

2. **What is the intended relationship** between the executor-level `DeleteTokensFromKvCache()` (ring buffer / StreamingLLM) and session-level checkpoint rewinding? Are these meant to be complementary or alternative approaches?

3. **Are there plans to expose** context pressure signals (e.g., ratio of `current_step / max_num_tokens`) at the `Conversation` API level, so applications can react before hitting hard limits?

4. **Will the `Conversation` class** eventually own context lifecycle management, or is this intended to remain an application-layer concern?

### Proposed Approach: Session-Level Context Shift

We've prototyped a context shift system built on top of the existing LiteRT-LM primitives that we believe would be a natural addition to the `Conversation` API. The design:

#### Configuration (added to `ConversationConfig`)

| Parameter | Default | Description |
|---|---|---|
| `context_shift_enabled` | `false` | Master switch |
| `context_shift_trigger_ratio` | `0.9` | Ratio of `current_step / max_num_tokens` that triggers a shift |
| `context_shift_target_ratio` | `0.8` | Target context usage after shift (must be ≤ trigger) |
| `context_shift_retain_recent_messages` | `8` | Number of recent conversation turns to replay after clearing |
| `context_shift_reset_on_exhaustion` | `true` | Destroy and recreate session if replay still exceeds budget |
| `context_shift_strategy` | `kReplayRecent` | Eviction policy (see below) |

#### Eviction Strategies

- **`kReplayRecent`** (default): Rewind to a saved anchor checkpoint (containing the prefilled system preface), then replay the N most recent messages. If replay still exceeds the target budget, iteratively drop the oldest replayed message until within budget.
- **`kDropAllButSystem`**: Rewind to the anchor checkpoint and discard all conversational turns, preserving only the system baseline.

#### Algorithm (`MaybeApplyContextShift`)

Called at the start of every `SendMessage` when a user message is present:

1. Query `session_->GetCurrentStep()` and compare to `trigger_step = max(1, max_context_tokens × trigger_ratio)`.
2. If below threshold, return (no action needed).
3. Build candidate replay messages from conversation history.
4. **Iterative budget-fitting loop:**
   - Rewind session to named `"context_shift_anchor_checkpoint"`.
   - Re-prefill replay messages via `session_->RunPrefill()`.
   - Check if `GetCurrentStep() ≤ target_step`. If yes, break.
   - If over budget, decrement replay count and retry with fewer messages.
5. **Exhaustion fallback:** If still over target with zero replay messages and `reset_on_exhaustion` is enabled, destroy the session entirely via `engine_.CreateSession()` and re-prefill the system preface from scratch.
6. Save a new anchor checkpoint for the next shift cycle.

#### Key Design Decisions

- **Checkpoint-based, not ring-buffer-based**: Uses `SaveCheckpoint` / `RewindToCheckpoint` rather than `DeleteTokensFromKvCache`. This preserves semantic coherence of the KV cache (no partial attention artifacts) and leverages existing session primitives.
- **Conversation-level, not executor-level**: The shift logic lives in `Conversation`, which has visibility into message boundaries, system prefaces, and history — information the executor layer doesn't have.
- **Graceful degradation**: The iterative shrink loop ensures the system always converges to a working state, even if it means dropping all history.
- **Backend-agnostic**: Automatically disables itself if the session backend returns `Unimplemented` for checkpoint operations.

### Why This Matters

- **Edge devices have hard context limits.** Unlike cloud deployments that can scale, on-device models like Gemma3-1B (4K context) will hit limits quickly in real conversations.
- **The primitives already exist.** Checkpointing, step querying, and KV cache management are all in place — this is a coordination layer, not new infrastructure.
- **Agentic workflows amplify the problem.** The recent push toward "agentic skills at the edge" (tool calling, function schemas) dramatically increases token consumption per turn, making context management even more urgent.
- **Developers need guidance.** Without a blessed approach, every integrator will build their own incompatible solution.

### References

- Existing KV cache work: #1807, #1800, #1680, #1601, #1550, #1491, #1452, #1422
- Session checkpoint API: `session.h` (`SaveCheckpoint`, `RewindToCheckpoint`, `GetCurrentStep`)
- StreamingLLM ring buffer: `llm_litert_compiled_model_cache_utils.cc`
- Conversation API: `runtime/conversation/conversation.h`

---

*We have a working implementation with tests covering the replay, drop-all, iterative shrink, and reset-on-exhaustion paths. Happy to contribute upstream if there's interest.*
