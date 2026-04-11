// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/execution_queue.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

// Configuration for the Conversation instance. This class is used to initialize
// the Conversation instance.
//
// To create a ConversationConfig, use ConversationConfig::CreateDefault() to
// create a default config, or use the ConversationConfig::Builder() to build a
// custom config.
//
// Note: Consider to remove ConversationConfig and use ConversationBuilder to
// build Conversation.
class ConversationConfig {
 public:
  // Boundary event for applying queued runtime policy updates.
  enum class PolicyApplyBoundary {
    kToolResult = 0,
    kTurnBoundary = 1,
  };

  // Policy for session-level context shift behavior.
  enum class ContextShiftStrategy {
    // Replays recent messages and may shrink replay window to fit budget.
    kReplayRecent = 0,
    // Drops all conversational turns and keeps only prefilled system baseline.
    kDropAllButSystem = 1,
  };

  // High-level memory management strategy selection.
  enum class MemoryStrategy {
    kHardResetReplayWindow = 0,
    kSummarizeProtectedTail = 1,
    kVirtualMemoryPaging = 2,
    kFactMemoryExtractionUpdate = 3,
    kSemanticCompressionConsolidationAdaptiveRetrieval = 4,
    kLearnedCompressionPolicy = 5,
    kIncrementalHierarchicalAggregation = 6,
    kActiveRecallSurpriseUpdate = 7,
    kContextualForgettingInterferenceManagement = 8,
    kTokenEfficientKvCacheManagement = 9,
    kReflectionMetacognitiveBuffering = 10,
    kSelfCorrectingFactGraph = 11,
    kSlowFastMemoryArchitecture = 12,
    kHeatBasedTieredMigration = 13,
    kContextQuarantineIsolatedScratchpads = 14,
    kMcpActiveMetadata = 15,
  };

  // Runtime memory policy used by hybrid profile+override control.
  enum class SafeBoundary {
    kTurnBoundary = 0,
    kToolResult = 1,
  };

  struct RuntimeMemoryPolicy {
    MemoryStrategy strategy = MemoryStrategy::kHardResetReplayWindow;
    bool context_shift_enabled = false;
    float context_shift_trigger_ratio = 0.9f;
    int context_shift_retain_recent_messages = 8;
    float context_shift_target_ratio = 0.8f;
    bool context_shift_reset_on_exhaustion = true;
    ContextShiftStrategy context_shift_strategy =
        ContextShiftStrategy::kReplayRecent;
    std::optional<std::string> profile_id = std::nullopt;
    std::optional<std::string> version = std::nullopt;
    std::optional<std::string> compatibility = std::nullopt;
    bool allow_runtime_tuning = true;
    SafeBoundary safe_boundary = SafeBoundary::kToolResult;
    std::optional<MemoryStrategy> shadow_strategy = std::nullopt;
    bool emit_transition_note = false;
  };

  // Creates a default ConversationConfig from the given Engine.
  // Args:
  // - `engine`: The Engine instance to be used for creating the default config.
  static absl::StatusOr<ConversationConfig> CreateDefault(const Engine& engine);

  // Converts a strategy string (e.g. "hard_reset_replay_window") to enum.
  static absl::StatusOr<MemoryStrategy> MemoryStrategyFromString(
      absl::string_view strategy_name);

  // Converts strategy enum to canonical profile string.
  static absl::string_view MemoryStrategyToString(MemoryStrategy strategy);

  // Converts a safe-boundary string (e.g. "tool_result") to enum.
  static absl::StatusOr<SafeBoundary> SafeBoundaryFromString(
      absl::string_view safe_boundary_name);

  // Converts safe-boundary enum to canonical profile string.
  static absl::string_view SafeBoundaryToString(SafeBoundary safe_boundary);

  // Parses a constrained YAML memory profile into runtime policy.
  static absl::StatusOr<RuntimeMemoryPolicy> ParseMemoryPolicyYaml(
      absl::string_view yaml_text);

  // Loads and parses a constrained YAML memory profile from file path.
  static absl::StatusOr<RuntimeMemoryPolicy> LoadMemoryPolicyYamlFile(
      absl::string_view yaml_file_path);

  // Returns the SessionConfig used for creating the ConversationConfig.
  const SessionConfig& GetSessionConfig() const { return session_config_; }

  // Returns the Preface used for creating the ConversationConfig.
  const Preface& GetPreface() const { return preface_; }

  // Returns the PromptTemplate used for creating the ConversationConfig.
  const PromptTemplate& GetPromptTemplate() const { return prompt_template_; }

  // Returns the DataProcessorConfig used for creating the ConversationConfig.
  const DataProcessorConfig& GetProcessorConfig() const {
    return processor_config_;
  }

  // Returns whether constrained decoding is enabled.
  bool constrained_decoding_enabled() const {
    return constrained_decoding_enabled_;
  }

  // Returns whether the preface should be prefilled when the Conversation is
  // created. This will make the first response faster, but take longer to
  // initialize.
  bool prefill_preface_on_init() const { return prefill_preface_on_init_; }

  // Returns the channels configured for the conversation.
  const std::vector<Channel>& GetChannels() const { return channels_; }

  // Returns whether to filter channel content from the KV cache.
  bool filter_channel_content_from_kv_cache() const {
    return filter_channel_content_from_kv_cache_;
  }

  // Returns whether context shift is enabled.
  bool context_shift_enabled() const { return context_shift_enabled_; }

  // Returns the context usage threshold ratio to trigger context shift.
  float context_shift_trigger_ratio() const {
    return context_shift_trigger_ratio_;
  }

  // Returns the number of recent messages to keep during context shift.
  int context_shift_retain_recent_messages() const {
    return context_shift_retain_recent_messages_;
  }

  // Returns the target ratio in (0, 1] for context usage after replay.
  float context_shift_target_ratio() const { return context_shift_target_ratio_; }

  // Returns whether to reset session when replay cannot fit target budget.
  bool context_shift_reset_on_exhaustion() const {
    return context_shift_reset_on_exhaustion_;
  }

  // Returns the context-shift strategy.
  ContextShiftStrategy context_shift_strategy() const {
    return context_shift_strategy_;
  }

  // Returns the selected high-level memory strategy.
  MemoryStrategy memory_strategy() const { return memory_strategy_; }

  // Returns whether runtime tuning is allowed for policy-update requests.
  bool allow_runtime_tuning() const { return allow_runtime_tuning_; }

  // Returns whether transition notes should be emitted for policy changes.
  bool emit_transition_note() const { return emit_transition_note_; }

  // Returns which boundary is used to apply queued control-plane policy updates.
  PolicyApplyBoundary policy_apply_boundary() const {
    return policy_apply_boundary_;
  }

  // Returns whether prefetch planner/install is enabled.
  bool prefetch_enabled() const { return prefetch_enabled_; }

  // Returns whether prefetch should run in shadow mode.
  bool prefetch_shadow_mode() const { return prefetch_shadow_mode_; }

  // Returns prefetch planning ratio in [0, 1].
  float prefetch_ratio() const { return prefetch_ratio_; }

  // Returns an equivalent runtime policy based on static config.
  RuntimeMemoryPolicy runtime_memory_policy() const {
    return RuntimeMemoryPolicy{
        .strategy = memory_strategy_,
        .context_shift_enabled = context_shift_enabled_,
        .context_shift_trigger_ratio = context_shift_trigger_ratio_,
        .context_shift_retain_recent_messages =
            context_shift_retain_recent_messages_,
        .context_shift_target_ratio = context_shift_target_ratio_,
        .context_shift_reset_on_exhaustion =
            context_shift_reset_on_exhaustion_,
        .context_shift_strategy = context_shift_strategy_,
        .profile_id = std::nullopt,
        .allow_runtime_tuning = allow_runtime_tuning_,
        .safe_boundary =
            policy_apply_boundary_ == PolicyApplyBoundary::kToolResult
                ? SafeBoundary::kToolResult
                : SafeBoundary::kTurnBoundary,
        .emit_transition_note = emit_transition_note_,
    };
  }

 public:
  // Builder class for ConversationConfig.
  //
  // Example usage:
  //   // Create a ConversationConfig instance using the Builder.
  //   ASSIGN_OR_RETURN(auto conversation_config,
  //                    ConversationConfig::Builder()
  //                        .SetEnableConstrainedDecoding(true)
  //                        .SetPrefillPrefaceOnInit(true)
  //                        .Build(*engine));
  class Builder {
   public:
    // Sets the SessionConfig to be used for creating the ConversationConfig.
    Builder& SetSessionConfig(const SessionConfig& session_config) {
      session_config_ = session_config;
      return *this;
    }

    // Sets the Preface for the conversation. The Preface provides
    // the initial background for the conversation, tool uses and extra
    // context for the conversation. If not provided, the conversation will
    // start with an empty Preface.
    Builder& SetPreface(const Preface& preface) {
      preface_ = preface;
      return *this;
    }

    // Sets the PromptTemplate instance to be used for the conversation. If
    // not provided, the conversation will use the template read from the model
    // metadata.
    Builder& SetOverwritePromptTemplate(
        const PromptTemplate& overwrite_prompt_template) {
      overwrite_prompt_template_ = overwrite_prompt_template;
      return *this;
    }

    // Sets the configuration for the model data processor. If not provided,
    // the default config for the model type's data processor will be used.
    // Most of the time, the users don't need to provide the data processor
    // config.
    Builder& SetOverwriteProcessorConfig(
        const DataProcessorConfig& overwrite_processor_config) {
      overwrite_processor_config_ = overwrite_processor_config;
      return *this;
    }

    // Sets whether to enable constrained decoding. If true, constrained
    // decoding will be used, primarily for function calling.
    Builder& SetEnableConstrainedDecoding(bool enable_constrained_decoding) {
      enable_constrained_decoding_ = enable_constrained_decoding;
      return *this;
    }

    // Sets whether to prefill the preface on init. If true, the preface will
    // be prefilled on init, which will make the first response faster, but
    // take longer to initialize.
    Builder& SetPrefillPrefaceOnInit(bool prefill_preface_on_init) {
      prefill_preface_on_init_ = prefill_preface_on_init;
      return *this;
    }

    // Sets the configuration for the constraint provider.
    Builder& SetConstraintProviderConfig(
        const ConstraintProviderConfig& constraint_provider_config) {
      constraint_provider_config_ = constraint_provider_config;
      return *this;
    }

    // Sets the channels for the conversation.
    Builder& SetChannels(const std::vector<Channel>& channels) {
      channels_ = channels;
      return *this;
    }

    // Sets whether to filter channel content from the KV cache. This is useful
    // when the model responds with "channel" content, e.g. thinking/reasoning
    // tokens, that should not be persisted in the KV cache.
    Builder& SetFilterChannelContentFromKvCache(
        bool filter_channel_content_from_kv_cache) {
      filter_channel_content_from_kv_cache_ =
          filter_channel_content_from_kv_cache;
      return *this;
    }

    // Enables session-level context shift when the context usage reaches
    // a configurable threshold.
    Builder& SetEnableContextShift(bool context_shift_enabled) {
      context_shift_enabled_ = context_shift_enabled;
      return *this;
    }

    // Sets the ratio in (0, 1] to trigger context shift based on
    // current_step / max_num_tokens.
    Builder& SetContextShiftTriggerRatio(float context_shift_trigger_ratio) {
      context_shift_trigger_ratio_ = context_shift_trigger_ratio;
      return *this;
    }

    // Sets how many most recent messages are replayed after a context shift.
    Builder& SetContextShiftRetainRecentMessages(
        int context_shift_retain_recent_messages) {
      context_shift_retain_recent_messages_ =
          context_shift_retain_recent_messages;
      return *this;
    }

    // Sets the target ratio in (0, 1] for context usage after replay.
    // If replayed history still exceeds this target, older messages are
    // progressively dropped until it fits the budget.
    Builder& SetContextShiftTargetRatio(float context_shift_target_ratio) {
      context_shift_target_ratio_ = context_shift_target_ratio;
      return *this;
    }

    // If true, reset and recreate the session when replay cannot fit the
    // context budget target. This mirrors Context Manager behavior where
    // the loop can restart with a fresh context.
    Builder& SetContextShiftResetOnExhaustion(
        bool context_shift_reset_on_exhaustion) {
      context_shift_reset_on_exhaustion_ = context_shift_reset_on_exhaustion;
      return *this;
    }

    // Sets how context shift should handle historical conversation context.
    Builder& SetContextShiftStrategy(ContextShiftStrategy strategy) {
      context_shift_strategy_ = strategy;
      return *this;
    }

    // Sets the high-level memory strategy.
    Builder& SetMemoryStrategy(MemoryStrategy strategy) {
      memory_strategy_ = strategy;
      return *this;
    }

    // Sets all runtime memory policy fields in one call.
    Builder& SetRuntimeMemoryPolicy(const RuntimeMemoryPolicy& policy) {
      memory_strategy_ = policy.strategy;
      context_shift_enabled_ = policy.context_shift_enabled;
      context_shift_trigger_ratio_ = policy.context_shift_trigger_ratio;
      context_shift_retain_recent_messages_ =
          policy.context_shift_retain_recent_messages;
      context_shift_target_ratio_ = policy.context_shift_target_ratio;
      context_shift_reset_on_exhaustion_ =
          policy.context_shift_reset_on_exhaustion;
      context_shift_strategy_ = policy.context_shift_strategy;
      allow_runtime_tuning_ = policy.allow_runtime_tuning;
      emit_transition_note_ = policy.emit_transition_note;
      policy_apply_boundary_ =
          policy.safe_boundary == SafeBoundary::kToolResult
              ? PolicyApplyBoundary::kToolResult
              : PolicyApplyBoundary::kTurnBoundary;
      return *this;
    }

    Builder& SetAllowRuntimeTuning(bool allow_runtime_tuning) {
      allow_runtime_tuning_ = allow_runtime_tuning;
      return *this;
    }

    Builder& SetEmitTransitionNote(bool emit_transition_note) {
      emit_transition_note_ = emit_transition_note;
      return *this;
    }

    Builder& SetPolicyApplyBoundary(PolicyApplyBoundary policy_apply_boundary) {
      policy_apply_boundary_ = policy_apply_boundary;
      return *this;
    }

    Builder& SetPrefetchEnabled(bool prefetch_enabled) {
      prefetch_enabled_ = prefetch_enabled;
      return *this;
    }

    Builder& SetPrefetchShadowMode(bool prefetch_shadow_mode) {
      prefetch_shadow_mode_ = prefetch_shadow_mode;
      return *this;
    }

    Builder& SetPrefetchRatio(float prefetch_ratio) {
      prefetch_ratio_ = prefetch_ratio;
      return *this;
    }

    absl::StatusOr<ConversationConfig> Build(const Engine& engine) {
      return ConversationConfig::CreateInternal(
          engine, session_config_, preface_, overwrite_prompt_template_,
          overwrite_processor_config_, enable_constrained_decoding_,
          prefill_preface_on_init_, constraint_provider_config_, channels_,
          filter_channel_content_from_kv_cache_, context_shift_enabled_,
          context_shift_trigger_ratio_, context_shift_retain_recent_messages_,
          context_shift_target_ratio_, context_shift_reset_on_exhaustion_,
          context_shift_strategy_, memory_strategy_, allow_runtime_tuning_,
          emit_transition_note_, policy_apply_boundary_, prefetch_enabled_,
          prefetch_shadow_mode_, prefetch_ratio_);
    }

    // Returns a unique pointer to a ConversationConfig.
    absl::StatusOr<std::unique_ptr<ConversationConfig>> BuildUnique(
        const Engine& engine) {
      ASSIGN_OR_RETURN(ConversationConfig config, Build(engine));
      return std::make_unique<ConversationConfig>(std::move(config));
    }

   private:
    SessionConfig session_config_ = SessionConfig::CreateDefault();
    std::optional<Preface> preface_;
    std::optional<PromptTemplate> overwrite_prompt_template_;
    std::optional<DataProcessorConfig> overwrite_processor_config_;
    bool enable_constrained_decoding_ = false;
    bool prefill_preface_on_init_ = false;
    std::optional<ConstraintProviderConfig> constraint_provider_config_;
    std::optional<std::vector<Channel>> channels_ = std::nullopt;
    bool filter_channel_content_from_kv_cache_ = false;
    bool context_shift_enabled_ = false;
    float context_shift_trigger_ratio_ = 0.9f;
    int context_shift_retain_recent_messages_ = 8;
    float context_shift_target_ratio_ = 0.8f;
    bool context_shift_reset_on_exhaustion_ = true;
    ContextShiftStrategy context_shift_strategy_ =
        ContextShiftStrategy::kReplayRecent;
    MemoryStrategy memory_strategy_ = MemoryStrategy::kHardResetReplayWindow;
    bool allow_runtime_tuning_ = true;
    bool emit_transition_note_ = false;
    PolicyApplyBoundary policy_apply_boundary_ = PolicyApplyBoundary::kTurnBoundary;
    bool prefetch_enabled_ = false;
    bool prefetch_shadow_mode_ = true;
    float prefetch_ratio_ = 0.8f;
  };

  // Returns the constrained decoding config.
  const std::optional<ConstraintProviderConfig>& constraint_provider_config()
      const {
    return constraint_provider_config_;
  }

 private:
  // Creates a ConversationConfig.
  // Args:
  // - `engine`: The Engine instance to be used to validate the SessionConfig.
  // - `session_config`: The SessionConfig to be used for creating the
  //     ConversationConfig.
  // - `preface`: Optional Preface for the conversation. The Preface provides
  //     the initial background for the conversation, tool uses and extra
  //     context for the conversation. If not provided, the conversation will
  //     start with an empty Preface.
  // - `overwrite_prompt_template`: Optional PromptTemplate instance to be used
  //     for the conversation. If not provided, the conversation will use the
  //     template read from the model metadata "jinja_prompt_template". If not
  //     provided, LiteRT-LM will try to generate a default one based on the llm
  //     model type.
  // - `overwrite_processor_config`: Optional configuration for the model data
  //     processor, if not provided, the default config for the model type's
  //     data processor will be used. Most of the time, the users don't need to
  //     provide the data processor config.
  // - `enable_constrained_decoding`: Whether to enable constrained decoding. If
  //     true, constrained decoding will be used, primarily for function
  //     calling.
  // - `prefill_preface_on_init`: Whether to prefill the preface on init. If
  //     true, the preface will be prefilled on init, which will make the first
  //     response faster, but take longer to initialize.
  // - `channels`: The channels configured for the conversation.
  // - `context_shift_enabled`: Whether to enable session-level context shift.
  // - `context_shift_trigger_ratio`: Trigger ratio in (0, 1] based on
  //   current_step / max_num_tokens.
  // - `context_shift_retain_recent_messages`: Number of recent messages to
  //   replay after a context shift.
  // - `context_shift_target_ratio`: Desired context-usage ratio after replay.
  // - `context_shift_reset_on_exhaustion`: Whether to reset/recreate session
  //   if replay cannot fit the target budget.
  // - `context_shift_strategy`: Strategy for what conversation context to keep
  //   during context shift.
  // - `memory_strategy`: High-level memory strategy identifier.
  static absl::StatusOr<ConversationConfig> CreateInternal(
      const Engine& engine, const SessionConfig& session_config,
      std::optional<Preface> preface = std::nullopt,
      std::optional<PromptTemplate> overwrite_prompt_template = std::nullopt,
      std::optional<DataProcessorConfig> overwrite_processor_config =
          std::nullopt,
      bool enable_constrained_decoding = false,
      bool prefill_preface_on_init = false,
      std::optional<ConstraintProviderConfig> constraint_provider_config =
          std::nullopt,
      std::optional<std::vector<Channel>> channels = std::nullopt,
      bool filter_channel_content_from_kv_cache = false,
      bool context_shift_enabled = false,
      float context_shift_trigger_ratio = 0.9f,
      int context_shift_retain_recent_messages = 8,
      float context_shift_target_ratio = 0.8f,
      bool context_shift_reset_on_exhaustion = true,
      ContextShiftStrategy context_shift_strategy =
          ContextShiftStrategy::kReplayRecent,
      MemoryStrategy memory_strategy = MemoryStrategy::kHardResetReplayWindow,
      bool allow_runtime_tuning = true,
      bool emit_transition_note = false,
      PolicyApplyBoundary policy_apply_boundary =
          PolicyApplyBoundary::kTurnBoundary,
      bool prefetch_enabled = false, bool prefetch_shadow_mode = true,
      float prefetch_ratio = 0.8f);

  explicit ConversationConfig(SessionConfig session_config, Preface preface,
                              PromptTemplate prompt_template,
                              DataProcessorConfig processor_config,
                              bool constrained_decoding_enabled = false,
                              bool prefill_preface_on_init = false,
                              std::optional<ConstraintProviderConfig>
                                  constraint_provider_config = std::nullopt,
                              std::vector<Channel> channels = {},
                              bool filter_channel_content_from_kv_cache = false,
                              bool context_shift_enabled = false,
                              float context_shift_trigger_ratio = 0.9f,
                              int context_shift_retain_recent_messages = 8,
                              float context_shift_target_ratio = 0.8f,
                              bool context_shift_reset_on_exhaustion = true,
                              ContextShiftStrategy context_shift_strategy =
                                  ContextShiftStrategy::kReplayRecent,
                              MemoryStrategy memory_strategy =
                                  MemoryStrategy::kHardResetReplayWindow,
                              bool allow_runtime_tuning = true,
                              bool emit_transition_note = false,
                              PolicyApplyBoundary policy_apply_boundary =
                                  PolicyApplyBoundary::kTurnBoundary,
                              bool prefetch_enabled = false,
                              bool prefetch_shadow_mode = true,
                              float prefetch_ratio = 0.8f)
      : session_config_(std::move(session_config)),
        preface_(std::move(preface)),
        prompt_template_(std::move(prompt_template)),
        processor_config_(std::move(processor_config)),
        constrained_decoding_enabled_(constrained_decoding_enabled),
        prefill_preface_on_init_(prefill_preface_on_init),
        constraint_provider_config_(std::move(constraint_provider_config)),
        channels_(std::move(channels)),
        filter_channel_content_from_kv_cache_(
            filter_channel_content_from_kv_cache),
        context_shift_enabled_(context_shift_enabled),
        context_shift_trigger_ratio_(context_shift_trigger_ratio),
        context_shift_retain_recent_messages_(
            context_shift_retain_recent_messages),
        context_shift_target_ratio_(context_shift_target_ratio),
        context_shift_reset_on_exhaustion_(
            context_shift_reset_on_exhaustion),
        context_shift_strategy_(context_shift_strategy),
        memory_strategy_(memory_strategy),
        allow_runtime_tuning_(allow_runtime_tuning),
        emit_transition_note_(emit_transition_note),
        policy_apply_boundary_(policy_apply_boundary),
        prefetch_enabled_(prefetch_enabled),
        prefetch_shadow_mode_(prefetch_shadow_mode),
        prefetch_ratio_(prefetch_ratio) {}

  SessionConfig session_config_;
  Preface preface_;
  PromptTemplate prompt_template_;
  DataProcessorConfig processor_config_;
  bool constrained_decoding_enabled_;
  bool prefill_preface_on_init_;
  std::optional<ConstraintProviderConfig> constraint_provider_config_;
  std::vector<Channel> channels_;
  bool filter_channel_content_from_kv_cache_;
  bool context_shift_enabled_;
  float context_shift_trigger_ratio_;
  int context_shift_retain_recent_messages_;
  float context_shift_target_ratio_;
  bool context_shift_reset_on_exhaustion_;
  ContextShiftStrategy context_shift_strategy_;
  MemoryStrategy memory_strategy_;
  bool allow_runtime_tuning_;
  bool emit_transition_note_;
  PolicyApplyBoundary policy_apply_boundary_;
  bool prefetch_enabled_;
  bool prefetch_shadow_mode_;
  float prefetch_ratio_;
};

// Runtime override payload for legacy policy-update requests.
struct ContextShiftRuntimePolicyOverride {
  std::optional<bool> context_shift_enabled = std::nullopt;
  std::optional<float> context_shift_trigger_ratio = std::nullopt;
  std::optional<int> context_shift_retain_recent_messages = std::nullopt;
  std::optional<float> context_shift_target_ratio = std::nullopt;
  std::optional<bool> context_shift_reset_on_exhaustion = std::nullopt;
  std::optional<ConversationConfig::ContextShiftStrategy> context_shift_strategy =
      std::nullopt;
};

// Legacy policy update request. This remains supported for compatibility and
// is synchronized with RuntimeMemoryPolicy state in Conversation.
struct ContextShiftPolicyUpdateRequest {
  int profile_schema_version = 1;
  int profile_compatibility_version = 1;
  ContextShiftRuntimePolicyOverride runtime_override;
  std::string reason;
};

// Optional arguments for sending a message to the LLM.
struct OptionalArgs {
  // Whether there is a pending message to be sent. If true, only the prefill
  // stage of LLM will be triggered, and the following decode stage will be
  // skipped. This is useful for the case where we need to append multiple
  // messages to the conversation, but only want to generate a response once.
  //
  // To also trigger the decode stage, set this field to false. Or to explicitly
  // trigger the decode stage only, set this field to false and send an empty
  // content message.
  //
  // Note: this option is only valid for model templates and
  // ModelDataProcessor that supports single turn prompt rendering.
  //
  // Example usages:
  //
  // Append multiple messages to the conversation without triggering the decode
  // stage.
  //
  // ASSERT_OK(conversation->SendMessage(
  //   JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
  //   {.has_pending_message = true}));
  //
  // ASSERT_OK(conversation->SendMessage(
  //   JsonMessage{{"role", "user"}, {"content", " This is a long message."}},
  //   {.has_pending_message = true}));
  //
  // By sending a message with has_pending_message set to false, the decode
  // stage will be triggered, and the decode result will be returned.
  //
  // ASSERT_OK(conversation->SendMessage(
  //   JsonMessage{{"role", "user"}, {"content", " This is the last message."}},
  //   {.has_pending_message = false}));
  //
  // Alternatively, send an empty message with has_pending_message set to false
  // to only trigger the decode stage.
  //
  // ASSERT_OK(conversation->SendMessage(
  //   JsonMessage{{"role", "user"}, {"content", " This is the last message."}},
  //   {.has_pending_message = true}));
  //
  // ASSERT_OK(conversation->SendMessage(
  //   JsonMessage{{"role", "user"}, {"content", ""}},
  //   {.has_pending_message = false}));
  bool has_pending_message = false;

  // The constraint to be used for constrained decoding.
  std::optional<ConstraintArg> decoding_constraint = std::nullopt;

  // The arguments for the model data processor. Most of the time, the users
  // don't need to provide this argument.
  std::optional<DataProcessorArguments> args = std::nullopt;

  // The maximum number of tokens to generate during decode.
  std::optional<int> max_output_tokens = std::nullopt;

  // The task group id for asynchronous tasks. If provided, the task
  // controller will be stored and can be cancelled by calling
  // `Conversation::CancelGroup(task_group_id)`.
  std::optional<std::string> task_group_id = std::nullopt;

  // The extra template context passed into PromptTemplateInput. This extra
  // context only applies to a single message and is merged with the extra
  // context provided in the Preface, overwriting existing keys.
  std::optional<nlohmann::ordered_json> extra_context = std::nullopt;

  // Optional runtime policy update request.
  std::optional<ContextShiftPolicyUpdateRequest> policy_update_request =
      std::nullopt;
};

// A multi-turn centric stateful Conversation API for high-level user
// interaction. Conversation maintains the history for users, so the users'
// messages will be used as the LLM context through the conversation.
//
// Conversation handles the complex data processing logic for Session usage,
// including:
// - Prompt template rendering.
// - Role-based messages handling.
// - Multimodal input processing.
// - History management.
// - Model-specific data processing.
//
// Example usage:
//
//   // Create an Engine instance.
//   ASSIGN_OR_RETURN(auto engine, Engine::Create(model_assets));
//
//   // Create a ConversationConfig instance from the Engine.
//   ASSIGN_OR_RETURN(auto conversation_config,
//                    ConversationConfig::CreateDefault(*engine));
//
//   // Create a Conversation instance.
//   ASSIGN_OR_RETURN(auto conversation,
//       Conversation::Create(*engine, conversation_config));
//
//   // Send a message to the LLM and returns the complete message.
//   ASSIGN_OR_RETURN(const Message message,
//                    conversation->SendMessage(JsonMessage{
//                        {"role", "user"}, {"content", "Hello world!"}}));
//
//   // Send a message to the LLM and process the asynchronous message results
//   // via the user_callback. The user_callback is a user-defined callback
//   // function that handles the message results.
//   EXPECT_OK(conversation->SendMessageAsync(
//       JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
//       [](absl::StatusOr<Message> message) {
//         // Handle the message results.
//         if (message.ok()) {
//           std::cout << "Message: " << std::endl;
//         }
//       });
//
class Conversation {
 public:
  // Structured transition record for runtime policy changes.
  struct PolicyTransitionRecord {
    enum class Action {
      kQueued = 0,
      kApplied = 1,
      kRejected = 2,
    };
    Action action;
    std::string old_policy;
    std::string new_policy;
    std::string boundary;
    std::string reason;
  };

  enum class PrefetchBuilderId {
    kReplayRecent = 0,
    kDropAllButSystem = 1,
    kSummarizeProtectedTail = 2,
    kQuarantineMerge = 3,
  };

  enum class PrefetchParityMode {
    kNotApplicable = 0,
    kStrictTokenParity = 1,
    kSemanticParity = 2,
  };

  struct PrefetchHistoryRange {
    int start_message_index = -1;
    int end_message_index_exclusive = -1;
  };

  enum class PrefetchReasonCode {
    kNone = 0,
    kPlanned = 1,
    kInstalled = 2,
    kStaleDiscarded = 3,
    kShadowSkipped = 4,
    kInstallFailed = 5,
    kFallback = 6,
    kPolicyUpdateQueued = 7,
    kPolicyChanged = 8,
    kHistoryRevisionChanged = 9,
    kRetainedSliceChanged = 10,
    kTargetStepExceeded = 11,
    kSupersededPlan = 12,
  };

  // Prefetch planner/install metrics.
  struct PrefetchMetrics {
    struct Dimensions {
      std::string profile_id;
      std::string strategy;
      std::string builder_id;
      std::string boundary;
      std::string model_type;
      std::string reason_code;
    };

    enum class Outcome {
      kPlanned = 0,
      kInstalled = 1,
      kStaleDiscarded = 2,
      kShadowSkipped = 3,
      kInstallFailed = 4,
      kFallback = 5,
    };

    struct Event {
      Outcome outcome = Outcome::kPlanned;
      Dimensions dimensions;
      PrefetchParityMode parity_mode = PrefetchParityMode::kNotApplicable;
      bool parity_mismatch = false;
      bool scaffold_only = false;
    };

    int planned_count = 0;
    int install_attempt_count = 0;
    int install_hit_count = 0;
    int stale_discard_count = 0;
    int shadow_skip_count = 0;
    int install_failure_count = 0;
    int fallback_count = 0;
    int parity_check_count = 0;
    int parity_mismatch_count = 0;
    double install_latency_ms_total = 0.0;
    double baseline_recompute_latency_ms_total = 0.0;
    Dimensions last_dimensions;
    PrefetchParityMode last_parity_mode = PrefetchParityMode::kNotApplicable;
    bool last_scaffold_only = false;
    std::vector<Event> events;
  };

  struct PrefetchEvent {
    std::optional<std::string> profile_id = std::nullopt;
    ConversationConfig::MemoryStrategy strategy =
        ConversationConfig::MemoryStrategy::kHardResetReplayWindow;
    PrefetchBuilderId builder_id = PrefetchBuilderId::kReplayRecent;
    std::optional<ConversationConfig::SafeBoundary> boundary = std::nullopt;
    std::string model_type = "unknown";
    PrefetchReasonCode reason_code = PrefetchReasonCode::kNone;
    PrefetchParityMode parity_mode = PrefetchParityMode::kNotApplicable;
    bool scaffold_only = false;
  };

  enum class PrefetchLifecycleState {
    kIdle = 0,
    kPlanned = 1,
    kComputing = 2,
    kReady = 3,
    kInstalled = 4,
    kDiscarded = 5,
  };

  enum class PrefetchInvalidationReason {
    kNone = 0,
    kPrefetchDisabled = 1,
    kContextShiftDisabled = 2,
    kBelowTrigger = 3,
    kExistingPlanStillUseful = 4,
    kPolicyUpdateQueued = 5,
    kPolicyChanged = 6,
    kHistoryRevisionChanged = 7,
    kRetainedSliceChanged = 8,
    kTargetStepExceeded = 9,
    kShadowMode = 10,
    kInstallFailed = 11,
    kSupersededPlan = 12,
  };

  struct PrefetchPlannerStateSnapshot {
    PrefetchLifecycleState lifecycle_state = PrefetchLifecycleState::kIdle;
    PrefetchInvalidationReason last_invalidation_reason =
        PrefetchInvalidationReason::kNone;
    uint64_t last_plan_history_revision = 0;
    size_t last_plan_policy_digest = 0;
    uint64_t active_plan_token = 0;
    int last_plan_source_step = 0;
    int last_successful_install_step = -1;
    float last_confidence_score = 0.0f;
  };

  enum class PrefetchInstallOutcome {
    kNoPendingPack = 0,
    kStaleDiscarded = 1,
    kShadowSkipped = 2,
    kInstalled = 3,
  };

  using NativeCacheCapabilities = Engine::Session::CacheOpCapabilities;
  using NativeCacheOpVerb = Engine::Session::CacheOpVerb;
  using NativeCachePinClass = Engine::Session::CachePinClass;
  using NativeCacheLogicalRole = Engine::Session::CacheLogicalRole;
  using NativeCacheFailureCode = Engine::Session::CacheOpFailureCode;

  struct NativeCacheStateSnapshot {
    bool attempted = false;
    bool committed = false;
    bool fallback_to_phase_b = false;  // Backward-compat alias of completed.
    bool fallback_attempted = false;
    bool fallback_completed = false;
    std::optional<NativeCacheFailureCode> last_failure_code = std::nullopt;
  };

  static absl::string_view NativeCacheOpVerbToString(NativeCacheOpVerb op_verb);
  static absl::string_view NativeCachePinClassToString(
      NativeCachePinClass pin_class);
  static absl::string_view NativeCacheLogicalRoleToString(
      NativeCacheLogicalRole logical_role);
  static absl::string_view NativeCacheFailureCodeToString(
      NativeCacheFailureCode failure_code);

  // Creates a Conversation instance from the the Engine and ConversationConfig.
  // Args:
  // - `engine`: The Engine instance to be used for creating the Conversation.
  // - `config`: The ConversationConfig instance to be used for creating the
  // Conversation.
  static absl::StatusOr<std::unique_ptr<Conversation>> Create(
      Engine& engine, const ConversationConfig& config);

  // Sends a message to the LLM and returns the complete message.
  // Args:
  // - `message`: The message to be sent to the LLM. If `message` is an array,
  //    each element will be treated as a separate message and be prefilled
  //    before generating the response.
  // - `optional_args`: The optional arguments for sending the message. See the
  //    definition of `OptionalArgs` for more details.
  // Returns :
  // - The complete message from the LLM.
  absl::StatusOr<Message> SendMessage(
      const Message& message, OptionalArgs optional_args = OptionalArgs());

  // Sends a message to the LLM and process the asynchronous message results via
  // the user_callback.
  // Args:
  // - `message`: The message to be sent to the LLM. If `message` is an array,
  //    each element will be treated as a separate message and be prefilled
  //    before generating the response.
  // - `user_callback`: The callback to receive the message events. The
  //    user_callback will be invoked in the following conditions:
  //    - On every new message chunk.
  //    - When the generation is complete, the user_callback will be invoked
  //      with an empty message.
  //    - When the generation is cancelled, the user_callback will be invoked
  //      with absl::CancelledError.
  //    - When an error occurs, the user_callback will be invoked with the error
  //      status.
  // - `optional_args`: The optional arguments for sending the message. See the
  //    definition of `OptionalArgs` for more details.
  // Returns :
  // - absl::OkStatus if the message is sent and processing successfully,
  //   otherwise the error status.
  absl::Status SendMessageAsync(
      const Message& message,
      absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
      OptionalArgs optional_args = OptionalArgs());

  // Scores the target text after the prefill process is done. This function
  // will run the decode process (with the existing context history) by feeding
  // in the provided target text tokens and fetch the decode output logits that
  // corresponds to the target text tokens. This is useful for running certain
  // scoring metrics, e.g. perplexity.
  // Note that the function will NOT update the conversation history or the
  // internal state of the Conversation. The existing context history will
  // remain the same after the function call.
  // Note also that the function will NOT apply any additional prompt template
  // to the target text as the goal is to get the score of the raw target text.
  // Args:
  //   - target_text: The target text to score.
  //   - returns: This function returns the score associated with each of the
  //     target texts. The scores are the log likelihood of the target text
  //     given the existing context history.
  absl::StatusOr<Responses> RunTextScoring(
      const std::vector<absl::string_view>& target_text,
      OptionalArgs optional_args = OptionalArgs());

  // Similar to the above RunTextScoring function, but this is a not blocking
  // call and the function will return right away. The processing status will
  // be signaled through the callback.
  absl::Status RunTextScoringAsync(
      const std::vector<absl::string_view>& target_text,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      OptionalArgs optional_args = OptionalArgs());

  // Returns the history of the conversation.
  // Note: the return value is a copy of the history, which may be expensive
  // for large history.
  std::vector<Message> GetHistory() const {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    return history_;
  }

  // Provides safe access to the conversation history without copying.
  // The provided visitor function is executed while the history mutex is held.
  // Args:
  // - visitor: The visitor function takes a const reference to the history
  //  vector.
  //
  // Example usage:
  //
  //   Message assistant_message;
  //   conversation->AccessHistory(
  //       [&assistant_message](const std::vector<Message>& history) {
  //         // Copy the last message to assistant_message. So we don't need to
  //         // copy the whole history, if we only need the last message.
  //         assistant_message = history.back();
  //       });
  void AccessHistory(absl::AnyInvocable<void(const std::vector<Message>&) const>
                         visitor) const {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    visitor(history_);
  }

  // Returns the configuration used for creating the Conversation.
  const ConversationConfig& GetConfig() const { return config_; }

  // Returns the benchmark info for the conversation. Under the hood, this
  // method triggers the benchmark info collection from the Session. Returns:
  // - The benchmark info for the conversation.
  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo();

  // Returns the mutable benchmark info for the conversation. Under the hood,
  // this method triggers the mutable benchmark info collection from the
  // Session. Returns:
  // - The mutable benchmark info for the conversation.
  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo();

  // Cancels the ongoing inference process, for asynchronous inference.
  // Note: the underlying Session is not rollbacked, so the message
  // from the user is actually sent to the LLM and processed for prefill.
  void CancelProcess();

  // Clones the conversation. The cloned conversation will be independent of the
  // original conversation, including the history, state, etc.
  //
  // Note that the cloned conversation will not clone the group_id of the
  // ongoing tasks.
  absl::StatusOr<std::unique_ptr<Conversation>> Clone();

  // Cancels all ongoing asynchronous tasks with the given task_group_id.
  // Args:
  // - `task_group_id`: The id of the task group to cancel.
  // Note: after the cancellation, there is no guarantee that the internal state
  // of the Conversation is intact and therefore it is recommended to not
  // continue using the Conversation after cancellation.
  void CancelGroup(absl::string_view task_group_id);

  // Applies a runtime memory policy override for this conversation instance.
  absl::Status SetRuntimeMemoryPolicy(
      const ConversationConfig::RuntimeMemoryPolicy& policy);

  // Applies runtime memory policy from YAML text.
  absl::Status SetRuntimeMemoryPolicyFromYaml(absl::string_view yaml_text);

  // Applies runtime memory policy from YAML file path.
  absl::Status SetRuntimeMemoryPolicyFromYamlFile(
      absl::string_view yaml_file_path);

  // Returns the active runtime policy (testing/debug helper).
  ConversationConfig::RuntimeMemoryPolicy GetActiveMemoryPolicyForTest() const {
    return GetActiveMemoryPolicy();
  }

  // Returns whether there is a queued runtime policy update (testing helper).
  int GetQueuedRuntimePolicyUpdateCountForTest() const {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    return pending_runtime_memory_policy_update_.has_value() ? 1 : 0;
  }

  // Returns structured policy transition records (testing helper).
  std::vector<PolicyTransitionRecord> GetPolicyTransitionRecordsForTest() const;

  // Returns deterministic transition note count (testing helper).
  int GetTransitionNoteCountForTest() const;

  // Returns active context shift strategy from control-plane runtime policy.
  ConversationConfig::ContextShiftStrategy
  GetActiveContextShiftStrategyForTest() const;

  // Returns queued control-plane policy update count (testing helper).
  int GetQueuedPolicyUpdateCountForTest() const;

  // Returns prefetch metrics snapshot (testing helper).
  PrefetchMetrics GetPrefetchMetricsForTest() const;

  // Returns planner state snapshot (testing helper).
  PrefetchPlannerStateSnapshot GetPrefetchPlannerStateForTest() const;

  // Returns immutable native-cache capabilities discovered at session start.
  NativeCacheCapabilities GetNativeCacheCapabilitiesForTest() const {
    return native_cache_capabilities_;
  }

  // Returns native cache-op state snapshot (testing helper).
  NativeCacheStateSnapshot GetNativeCacheStateForTest() const;

  // Waits until planner reaches the given lifecycle state (testing helper).
  bool WaitForPrefetchPlannerStateForTest(
      PrefetchLifecycleState desired_state,
      absl::Duration timeout = absl::Seconds(10)) const;

  // Triggers planning/install paths directly for focused unit tests.
  void MaybePlanPrefetchPackForTest(int current_step) {
    MaybePlanPrefetchPack(current_step);
  }
  absl::StatusOr<PrefetchInstallOutcome> TryInstallPrefetchPackForTest(
      int target_step) {
    return TryInstallPrefetchPackIfValid(target_step);
  }

  // Mutates one history entry for testing staleness checks.
  absl::Status ReplaceHistoryMessageForTest(int history_index,
                                            const Message& replacement,
                                            bool increment_revision);

 private:
  enum class BoundaryEvent {
    kToolResult = 0,
    kTurnBoundary = 1,
    kUnknown = 2,
  };

  struct ContextShiftRuntimePolicy {
    bool context_shift_enabled = false;
    float context_shift_trigger_ratio = 0.9f;
    int context_shift_retain_recent_messages = 8;
    float context_shift_target_ratio = 0.8f;
    bool context_shift_reset_on_exhaustion = true;
    ConversationConfig::ContextShiftStrategy context_shift_strategy =
        ConversationConfig::ContextShiftStrategy::kReplayRecent;
  };

  struct PendingPolicyUpdate {
    ContextShiftPolicyUpdateRequest request;
    BoundaryEvent boundary;
  };

  struct PrefetchReplayPack {
    uint64_t plan_token = 0;
    int source_checkpoint_step = 0;
    size_t history_watermark = 0;
    uint64_t history_revision = 0;
    PrefetchBuilderId builder_id = PrefetchBuilderId::kReplayRecent;
    std::vector<PrefetchHistoryRange> retained_ranges;
    std::vector<PrefetchHistoryRange> protected_ranges;
    bool summary_anchor_present = false;
    bool scaffold_only = false;
    PrefetchParityMode parity_mode = PrefetchParityMode::kNotApplicable;
    int retained_start_index = -1;
    int retained_end_index_exclusive = -1;
    size_t retained_history_digest = 0;
    size_t policy_digest = 0;
    size_t artifact_identity_digest = 0;
    float target_ratio = 0.0f;
    float confidence_score = 0.0f;
    int planned_target_step = 0;
    ConversationConfig::ContextShiftStrategy strategy =
        ConversationConfig::ContextShiftStrategy::kReplayRecent;
    size_t validity_hash = 0;
    std::vector<InputData> replay_inputs;
  };

  struct PrefetchPlannerState {
    PrefetchLifecycleState lifecycle_state = PrefetchLifecycleState::kIdle;
    PrefetchInvalidationReason last_invalidation_reason =
        PrefetchInvalidationReason::kNone;
    uint64_t last_plan_history_revision = 0;
    size_t last_plan_policy_digest = 0;
    uint64_t active_plan_token = 0;
    uint64_t next_plan_token = 0;
    int last_plan_source_step = 0;
    int last_observed_step = 0;
    int last_successful_install_step = -1;
    float last_confidence_score = 0.0f;
  };

  struct NativeCacheState {
    bool attempted = false;
    bool committed = false;
    bool fallback_attempted = false;
    bool fallback_completed = false;
    std::optional<NativeCacheFailureCode> last_failure_code = std::nullopt;
  };

  explicit Conversation(
      Engine& engine, std::unique_ptr<Engine::Session> session,
      std::unique_ptr<ModelDataProcessor> model_data_processor, Preface preface,
      PromptTemplate prompt_template, ConversationConfig config,
      std::unique_ptr<ConstraintProvider> constraint_provider = nullptr)
      : engine_(engine),
        model_data_processor_(std::move(model_data_processor)),
        preface_(preface),
        prompt_template_(std::move(prompt_template)),
        config_(config),
        constraint_provider_(std::move(constraint_provider)),
        session_(std::move(session)) {}

  absl::StatusOr<std::string> GetSingleTurnText(
      const Message& message, const OptionalArgs& optional_args);

  absl::StatusOr<std::string> GetSingleTurnTextFromFullHistory(
      const JsonMessage& json_message, const OptionalArgs& optional_args);

  absl::StatusOr<std::string> GetSingleTurnTextFromSingleTurnTemplate(
      const JsonMessage& json_message, const OptionalArgs& optional_args);

  absl::StatusOr<DecodeConfig> CreateDecodeConfig(
      std::optional<ConstraintArg> decoding_constraint = std::nullopt,
      std::optional<int> max_output_tokens = std::nullopt);

  // Adds a task controller to the task_controllers_ map if task_group_id is
  // provided.
  // Args:
  // - `task_group_id`: The id of the task group to add the controller to.
  // - `task_controller`: The task controller to add.
  void AddTaskController(
      const std::optional<std::string>& task_group_id,
      std::unique_ptr<Engine::Session::TaskController> task_controller);

  // Returns the prefill text for the given messages.
  //
  // The prefill text is obtained by taking the difference between the rendered
  // string when the template context contains only the old message and the
  // rendered string when the template context contains both the new and old
  // messages.
  //
  // Args:
  // - `old_messages`: The old messages that have already been prefilled.
  // - `new_messages`: The new messages to be prefilled.
  // - `optional_args`: The optional arguments for template rendering.
  absl::StatusOr<std::string> GetPrefillTextForMessages(
      absl::Span<const Message> old_messages,
      absl::Span<const Message> new_messages,
      const OptionalArgs& optional_args = OptionalArgs());

  // Returns the input data vector for the given messages.
  //
  // Gets the prefill text for `new_messages` and converts it to an input data
  // vector for `Session::RunPrefill`.
  //
  // Args:
  // - `old_messages`: The old messages that have already been prefilled.
  // - `new_messages`: The new messages to be prefilled.
  // - `optional_args`: The optional arguments for template rendering.
  absl::StatusOr<std::vector<InputData>> GetInputDataVectorForMessages(
      absl::Span<const Message> old_messages,
      absl::Span<const Message> new_messages,
      const OptionalArgs& optional_args = OptionalArgs());

  // Same as above, but uses a caller-provided processor instance. This is used
  // by background prefetch planning to avoid sharing mutable processor state
  // across threads.
  absl::StatusOr<std::vector<InputData>> GetInputDataVectorForMessagesWithProcessor(
      const ModelDataProcessor& processor, absl::Span<const Message> old_messages,
      absl::Span<const Message> new_messages,
      const OptionalArgs& optional_args = OptionalArgs()) const;

  // Rewinds the session to the checkpoint after the most recent channel content
  // and return the input data vector for all messages from that point onward.
  absl::StatusOr<std::vector<InputData>> RewindAndGetInputDataVector();

  // Triggers session-level context shift and replays recent messages when
  // context usage reaches the configured threshold.
  absl::Status MaybeApplyContextShift();
  absl::StatusOr<bool> TryApplyNativeContextShift(int current_step,
                                                  int target_step);
  void RecordNativeCacheState(
      bool attempted, bool committed, bool fallback_attempted,
      bool fallback_completed,
      std::optional<NativeCacheFailureCode> last_failure_code);
  void MarkNativeFallbackAttemptedIfNeeded();
  void MarkNativeFallbackCompletedIfNeeded();
  static bool ShouldFallbackToPhaseB(NativeCacheFailureCode failure_code);

  // Schedules background prefetch planning after a safe boundary.
  void MaybeSchedulePrefetchPlanAfterBoundary();

  // Cancels any queued prefetch planning task. Running tasks cannot be
  // interrupted and must self-discard via plan-token guards.
  void CancelQueuedPrefetchPlan(PrefetchInvalidationReason reason);

  PrefetchBuilderId SelectPrefetchBuilderId(
      const ConversationConfig::RuntimeMemoryPolicy& policy) const;
  std::vector<PrefetchHistoryRange> BuildRetainedHistoryRanges(
      PrefetchBuilderId builder_id,
      const ContextShiftRuntimePolicy& policy_snapshot,
      int* retained_start_index, int* retained_end_index_exclusive,
      std::vector<Message>* candidate_messages) const;
  std::vector<PrefetchHistoryRange> BuildProtectedHistoryRanges(
      PrefetchBuilderId builder_id,
      const std::vector<PrefetchHistoryRange>& retained_ranges) const;
  PrefetchParityMode SelectPrefetchParityMode(
      PrefetchBuilderId builder_id) const;
  size_t ComputePrefetchArtifactIdentityDigest(
      PrefetchBuilderId builder_id,
      absl::Span<const PrefetchHistoryRange> retained_ranges,
      absl::Span<const PrefetchHistoryRange> protected_ranges,
      bool summary_anchor_present, bool scaffold_only,
      PrefetchParityMode parity_mode,
      const ContextShiftRuntimePolicy& policy) const;
  static absl::string_view PrefetchBuilderIdToString(PrefetchBuilderId builder_id);
  static absl::string_view PrefetchParityModeToString(
      PrefetchParityMode parity_mode);
  static absl::string_view PrefetchReasonCodeToString(
      PrefetchReasonCode reason_code);
  static bool IsValidHistoryRange(const PrefetchHistoryRange& range);
  PrefetchMetrics::Dimensions BuildPrefetchMetricDimensions(
      const PrefetchReplayPack* pack, PrefetchReasonCode reason_code,
      std::optional<ConversationConfig::SafeBoundary> boundary) const;
  void RecordPrefetchEvent(
      const PrefetchReplayPack* pack, PrefetchReasonCode reason_code,
      std::optional<ConversationConfig::SafeBoundary> boundary = std::nullopt,
      std::optional<PrefetchInstallOutcome> install_outcome = std::nullopt);

  // Clones the processor state for background prefetch planning.
  absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
  CloneModelDataProcessorForPrefetch() const;

  // Background task entry point for building a replay pack.
  void RunPrefetchPlanJob(int current_step, uint64_t plan_token,
                          std::unique_ptr<ModelDataProcessor> processor);

  // Plans/installs prefetch replay packs and tracks telemetry.
  void MaybePlanPrefetchPack(int current_step);
  absl::StatusOr<PrefetchInstallOutcome> TryInstallPrefetchPackIfValid(
      int target_step);
  float ComputePrefetchConfidenceScore(
      int current_step, int step_delta, int last_successful_install_step,
      const ContextShiftRuntimePolicy& policy) const;
  size_t ComputePrefetchPolicyDigest(
      const ContextShiftRuntimePolicy& policy) const;
  size_t ComputePrefetchValidityHash() const;
  void RecordPrefetchMetric(absl::FunctionRef<void(PrefetchMetrics&)> updater);

  // Computes a deterministic digest over a history slice for staleness checks.
  size_t ComputeRetainedHistoryDigest(
      int retained_start_index, int retained_end_index_exclusive,
      ConversationConfig::ContextShiftStrategy strategy) const;

  // Returns currently active runtime memory policy (override or config).
  ConversationConfig::RuntimeMemoryPolicy GetActiveMemoryPolicy() const;

  // Applies queued policy updates at safe boundary if any.
  absl::Status ApplyPendingRuntimeMemoryPolicyAtSafeBoundary(
      ConversationConfig::SafeBoundary boundary);

  // Applies a policy immediately. Caller must ensure boundary safety.
  absl::Status ApplyRuntimeMemoryPolicyNow(
      const ConversationConfig::RuntimeMemoryPolicy& policy);

  // Anchors context with short replay and optional transition note.
  absl::Status AnchorContextForPolicyTransition(
      const ConversationConfig::RuntimeMemoryPolicy& policy);

  // Queues a policy update to be applied at a safe boundary.
  void QueueRuntimeMemoryPolicyUpdate(
      const ConversationConfig::RuntimeMemoryPolicy& policy);

  // Control-plane helpers for boundary detection + transitions.
  BoundaryEvent DetectBoundaryEvent(const nlohmann::ordered_json& json_msg) const;
  absl::Status RequestPolicyUpdate(const ContextShiftPolicyUpdateRequest& request,
                                   BoundaryEvent boundary);
  absl::Status MaybeApplyQueuedPolicyAtBoundary(BoundaryEvent boundary);
  absl::Status ValidatePolicyUpdateRequest(
      const ContextShiftPolicyUpdateRequest& request,
      const ContextShiftRuntimePolicy& current_policy) const;
  ContextShiftRuntimePolicy ResolveEffectivePolicy(
      const ContextShiftRuntimePolicy& current_policy,
      const ContextShiftRuntimePolicyOverride& runtime_override) const;
  std::string SerializePolicy(const ContextShiftRuntimePolicy& policy) const;
  void RecordPolicyTransition(PolicyTransitionRecord::Action action,
                              BoundaryEvent boundary, absl::string_view reason,
                              const ContextShiftRuntimePolicy& old_policy,
                              const ContextShiftRuntimePolicy& new_policy);
  void MaybeEmitTransitionNote(BoundaryEvent boundary,
                               const ContextShiftRuntimePolicy& old_policy,
                               const ContextShiftRuntimePolicy& new_policy);

  // Tracks model-turn active state for atomic policy transitions.
  void SetModelTurnActive(bool active);
  bool IsModelTurnActive() const;

  // Prefills the configured preface on the current session when enabled.
  absl::Status PrefillPrefaceIfConfigured();

  // Keep a reference to the creator engine to enable access to the shared
  // resources that might be required for features like cloning.
  Engine& engine_;
  std::unique_ptr<ModelDataProcessor> model_data_processor_;
  Preface preface_;
  PromptTemplate prompt_template_;
  // The constraint is currently created from the tools defined in the preface,
  // if any.
  std::unique_ptr<Constraint> constraint_;
  const ConversationConfig config_;
  std::unique_ptr<ConstraintProvider> constraint_provider_ = nullptr;
  mutable absl::Mutex history_mutex_;
  std::vector<Message> history_ ABSL_GUARDED_BY(history_mutex_);
  uint64_t history_revision_ ABSL_GUARDED_BY(history_mutex_) = 0;

  // Whether the current conversation is in message appending state.
  bool is_appending_message_ = false;

  // Mutex protecting runtime policy queue/state and prefetch metadata.
  mutable absl::Mutex policy_mutex_;

  // Active effective context-shift policy.
  ContextShiftRuntimePolicy active_context_shift_policy_
      ABSL_GUARDED_BY(policy_mutex_);

  // Pending policy updates queue.
  std::vector<PendingPolicyUpdate> pending_policy_updates_
      ABSL_GUARDED_BY(policy_mutex_);

  // Structured transition records.
  std::vector<PolicyTransitionRecord> policy_transition_records_
      ABSL_GUARDED_BY(policy_mutex_);

  // Deterministic internal transition notes.
  std::vector<std::string> transition_notes_ ABSL_GUARDED_BY(policy_mutex_);

  // Prefetch replay pack + metrics.
  std::optional<PrefetchReplayPack> pending_prefetch_pack_
      ABSL_GUARDED_BY(policy_mutex_);
  std::optional<int> queued_prefetch_task_id_ ABSL_GUARDED_BY(policy_mutex_);
  PrefetchPlannerState prefetch_planner_state_ ABSL_GUARDED_BY(policy_mutex_);
  PrefetchMetrics prefetch_metrics_ ABSL_GUARDED_BY(policy_mutex_);
  NativeCacheState native_cache_state_ ABSL_GUARDED_BY(policy_mutex_);

  // Mutex for task_controllers_.
  mutable absl::Mutex task_controllers_mutex_;
  // Map of task group id to task controllers.
  absl::flat_hash_map<
      std::string,
      std::vector<std::unique_ptr<Engine::Session::TaskController>>>
      task_controllers_ ABSL_GUARDED_BY(task_controllers_mutex_);

  // Declare the session after model_data_processor_ and other members it
  // depends on so that the session is destroyed before them. This is to avoid
  // memory corruption and null-pointer deference issues.
  std::unique_ptr<Engine::Session> session_;
  NativeCacheCapabilities native_cache_capabilities_;

  // Background planner queue. Declare after session_ so it is destroyed first,
  // which ensures all queued work finishes before other runtime state tears
  // down.
  std::unique_ptr<ExecutionQueue> prefetch_execution_queue_;

  // Whether checkpointing and rewinding are supported by the session.

  // Assumed to be true initially but on the first error from SaveCheckpoint,
  // will be set to false.  Rewinding is supported by SessionBasic but not by
  // SessionAdvanced.
  //
  //  TODO(b/494425377): Support rewinding in SessionAdvanced and remove
  //  session_checkpoint_supported_.
  bool session_checkpoint_supported_ = true;

  // Whether context-shift checkpointing and rewinding are supported by
  // the underlying session implementation.
  bool context_shift_supported_ = true;

  // The index of the message you have to rewind to in order to remove channel
  // content from the KV cache. nullopt means no rewind is needed.
  std::optional<int> checkpoint_message_index_ = std::nullopt;

  // Max number of tokens supported by the model context.
  int max_context_tokens_ = 0;

  // Runtime override for memory policy that can be changed between turns.
  mutable absl::Mutex memory_policy_mutex_;
  std::optional<ConversationConfig::RuntimeMemoryPolicy>
      runtime_memory_policy_override_ ABSL_GUARDED_BY(memory_policy_mutex_);
  std::optional<ConversationConfig::RuntimeMemoryPolicy>
      pending_runtime_memory_policy_update_
          ABSL_GUARDED_BY(memory_policy_mutex_);
  bool model_turn_active_ ABSL_GUARDED_BY(memory_policy_mutex_) = false;
  bool policy_transition_blocked_ ABSL_GUARDED_BY(memory_policy_mutex_) = false;
};
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
