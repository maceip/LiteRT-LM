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

#include "runtime/conversation/conversation.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/channel_util.h"
#include "runtime/conversation/internal_callback_util.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/conversation/prompt_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/execution_queue.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/model_type_utils.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

constexpr absl::string_view kRoleKey = "role";
constexpr absl::string_view kUser = "user";
constexpr absl::string_view kTool = "tool";
constexpr absl::string_view kChannelsKey = "channels";
constexpr absl::string_view kChannelContentCheckpoint =
    "channel_content_checkpoint";
constexpr absl::string_view kContextShiftAnchorCheckpoint =
    "context_shift_anchor_checkpoint";

bool IsEmptyInputError(const absl::Status& status) {
  return absl::IsInvalidArgument(status) &&
         absl::StrContains(status.message(), "Input is empty");
}

// Ignores the invalid argument error when Session Prefill is called with empty
// input.
absl::Status IgnoreEmptyInputError(const absl::Status& status) {
  return IsEmptyInputError(status) ? absl::OkStatus() : status;
}

bool IsEmptyPreface(const Preface& preface) {
  auto json_preface = std::get<JsonPreface>(preface);
  return (json_preface.messages.is_null() || json_preface.messages.empty()) &&
         (json_preface.tools.is_null() || json_preface.tools.empty()) &&
         (json_preface.extra_context.is_null() ||
          json_preface.extra_context.empty());
}

bool IsUserMessage(const nlohmann::ordered_json& json_msg) {
  return json_msg.contains(kRoleKey) && json_msg[kRoleKey].is_string() &&
         json_msg[kRoleKey].get<absl::string_view>() == kUser;
}

bool ContainsUserMessage(const nlohmann::ordered_json& json_msg) {
  if (json_msg.is_array()) {
    for (const auto& message : json_msg) {
      if (message.is_object() && IsUserMessage(message)) {
        return true;
      }
    }
    return false;
  }
  return json_msg.is_object() && IsUserMessage(json_msg);
}

bool IsToolMessage(const nlohmann::ordered_json& json_msg) {
  return json_msg.contains(kRoleKey) && json_msg[kRoleKey].is_string() &&
         json_msg[kRoleKey].get<absl::string_view>() == kTool;
}

bool ContainsToolMessage(const nlohmann::ordered_json& json_msg) {
  if (json_msg.is_array()) {
    for (const auto& message : json_msg) {
      if (message.is_object() && IsToolMessage(message)) {
        return true;
      }
    }
    return false;
  }
  return json_msg.is_object() && IsToolMessage(json_msg);
}

Message MaybeStripChannelContentFromMessage(
    const Message& message, bool strip_channel_content) {
  if (!strip_channel_content ||
      !std::holds_alternative<nlohmann::ordered_json>(message)) {
    return message;
  }
  nlohmann::ordered_json json_message =
      std::get<nlohmann::ordered_json>(message);
  if (json_message.is_object()) {
    json_message.erase(std::string(kChannelsKey));
  }
  return json_message;
}

std::string Trim(std::string input) {
  const auto begin = input.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = input.find_last_not_of(" \t\r\n");
  return input.substr(begin, end - begin + 1);
}

std::string NormalizeStrategyName(absl::string_view strategy_name) {
  std::string out;
  out.reserve(strategy_name.size());
  bool prev_underscore = false;
  for (char c : strategy_name) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      out.push_back(static_cast<char>(
          std::tolower(static_cast<unsigned char>(c))));
      prev_underscore = false;
      continue;
    }
    if (!prev_underscore) {
      out.push_back('_');
      prev_underscore = true;
    }
  }
  while (!out.empty() && out.front() == '_') out.erase(out.begin());
  while (!out.empty() && out.back() == '_') out.pop_back();
  return out;
}

template <typename T>
void HashCombine(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

absl::Status ValidateRuntimeMemoryPolicy(
    const ConversationConfig::RuntimeMemoryPolicy& policy) {
  if (policy.context_shift_trigger_ratio <= 0.0f ||
      policy.context_shift_trigger_ratio > 1.0f) {
    return absl::InvalidArgumentError(
        "context_shift_trigger_ratio must be in (0, 1].");
  }
  if (policy.context_shift_retain_recent_messages < 0) {
    return absl::InvalidArgumentError(
        "context_shift_retain_recent_messages must be >= 0.");
  }
  if (policy.context_shift_target_ratio <= 0.0f ||
      policy.context_shift_target_ratio > 1.0f) {
    return absl::InvalidArgumentError(
        "context_shift_target_ratio must be in (0, 1].");
  }
  if (policy.context_shift_target_ratio > policy.context_shift_trigger_ratio) {
    return absl::InvalidArgumentError(
        "context_shift_target_ratio must be <= context_shift_trigger_ratio.");
  }
  switch (policy.context_shift_strategy) {
    case ConversationConfig::ContextShiftStrategy::kReplayRecent:
    case ConversationConfig::ContextShiftStrategy::kDropAllButSystem:
      break;
  }
  switch (policy.strategy) {
    case ConversationConfig::MemoryStrategy::kHardResetReplayWindow:
    case ConversationConfig::MemoryStrategy::kSummarizeProtectedTail:
    case ConversationConfig::MemoryStrategy::kVirtualMemoryPaging:
    case ConversationConfig::MemoryStrategy::kFactMemoryExtractionUpdate:
    case ConversationConfig::MemoryStrategy::
        kSemanticCompressionConsolidationAdaptiveRetrieval:
    case ConversationConfig::MemoryStrategy::kLearnedCompressionPolicy:
    case ConversationConfig::MemoryStrategy::
        kIncrementalHierarchicalAggregation:
    case ConversationConfig::MemoryStrategy::kActiveRecallSurpriseUpdate:
    case ConversationConfig::MemoryStrategy::
        kContextualForgettingInterferenceManagement:
    case ConversationConfig::MemoryStrategy::kTokenEfficientKvCacheManagement:
    case ConversationConfig::MemoryStrategy::kReflectionMetacognitiveBuffering:
    case ConversationConfig::MemoryStrategy::kSelfCorrectingFactGraph:
    case ConversationConfig::MemoryStrategy::kSlowFastMemoryArchitecture:
    case ConversationConfig::MemoryStrategy::kHeatBasedTieredMigration:
    case ConversationConfig::MemoryStrategy::
        kContextQuarantineIsolatedScratchpads:
    case ConversationConfig::MemoryStrategy::kMcpActiveMetadata:
      break;
  }
  return absl::OkStatus();
}

bool TryParseBool(absl::string_view text, bool* out) {
  std::string lowered = NormalizeStrategyName(text);
  if (lowered == "true" || lowered == "yes" || lowered == "on" ||
      lowered == "1") {
    *out = true;
    return true;
  }
  if (lowered == "false" || lowered == "no" || lowered == "off" ||
      lowered == "0") {
    *out = false;
    return true;
  }
  return false;
}

bool IsSupportedPolicyFieldValue(absl::string_view value) {
  const std::string normalized = NormalizeStrategyName(value);
  return normalized == "1" || normalized == "v1" ||
         normalized == "phase_a_v1";
}

absl::Status ValidateRuntimePolicyVersionCompatibility(
    const ConversationConfig::RuntimeMemoryPolicy& policy) {
  if (policy.version.has_value() &&
      !IsSupportedPolicyFieldValue(*policy.version)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported runtime policy version: ",
                     *policy.version, ". Expected v1-compatible value."));
  }
  if (policy.compatibility.has_value() &&
      !IsSupportedPolicyFieldValue(*policy.compatibility)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported runtime policy compatibility: ",
                     *policy.compatibility, ". Expected v1-compatible value."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::string>> GetYamlValue(
    const absl::flat_hash_map<std::string, std::string>& kv,
    std::initializer_list<absl::string_view> keys) {
  for (absl::string_view key : keys) {
    auto it = kv.find(std::string(key));
    if (it != kv.end()) {
      return it->second;
    }
  }
  return std::nullopt;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ParseConstrainedYaml(absl::string_view yaml_text) {
  absl::flat_hash_map<std::string, std::string> out;
  std::vector<std::pair<int, std::string>> key_stack;
  std::stringstream ss{std::string(yaml_text)};
  std::string line;
  int line_number = 0;
  while (std::getline(ss, line)) {
    ++line_number;
    const auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }
    int indent = 0;
    while (indent < static_cast<int>(line.size()) && line[indent] == ' ') {
      ++indent;
    }
    const auto colon_pos = trimmed.find(':');
    if (colon_pos == std::string::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid YAML line ", line_number, ": missing ':'"));
    }
    std::string key = Trim(trimmed.substr(0, colon_pos));
    std::string value = Trim(trimmed.substr(colon_pos + 1));
    if (key.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid YAML line ", line_number, ": empty key"));
    }

    while (!key_stack.empty() && indent <= key_stack.back().first) {
      key_stack.pop_back();
    }

    if (value.empty()) {
      key_stack.push_back({indent, key});
      continue;
    }

    std::string dotted_key;
    for (const auto& [_, prefix] : key_stack) {
      absl::StrAppend(&dotted_key, prefix, ".");
    }
    absl::StrAppend(&dotted_key, key);
    out[dotted_key] = value;
    out[key] = value;
  }
  return out;
}

}  // namespace

absl::StatusOr<ConversationConfig> ConversationConfig::CreateDefault(
    const Engine& engine) {
  return ConversationConfig::Builder().Build(engine);
}

absl::StatusOr<ConversationConfig::MemoryStrategy>
ConversationConfig::MemoryStrategyFromString(absl::string_view strategy_name) {
  const std::string normalized = NormalizeStrategyName(strategy_name);
  using S = ConversationConfig::MemoryStrategy;
  static const absl::flat_hash_map<std::string, MemoryStrategy> kMap = {
      {"hard_reset_replay_window", S::kHardResetReplayWindow},
      {"step_style_context_manager", S::kHardResetReplayWindow},
      {"summarize_protected_tail", S::kSummarizeProtectedTail},
      {"hermes_style", S::kSummarizeProtectedTail},
      {"virtual_memory_paging", S::kVirtualMemoryPaging},
      {"memgpt", S::kVirtualMemoryPaging},
      {"fact_memory_extraction_update", S::kFactMemoryExtractionUpdate},
      {"mem0", S::kFactMemoryExtractionUpdate},
      {"mem0_g", S::kFactMemoryExtractionUpdate},
      {"semantic_compression_consolidation_adaptive_retrieval",
       S::kSemanticCompressionConsolidationAdaptiveRetrieval},
      {"simplemem", S::kSemanticCompressionConsolidationAdaptiveRetrieval},
      {"learned_compression_policy", S::kLearnedCompressionPolicy},
      {"acon", S::kLearnedCompressionPolicy},
      {"incremental_hierarchical_aggregation",
       S::kIncrementalHierarchicalAggregation},
      {"raptor", S::kIncrementalHierarchicalAggregation},
      {"active_recall_surprise_update", S::kActiveRecallSurpriseUpdate},
      {"swirl", S::kActiveRecallSurpriseUpdate},
      {"contextual_forgetting_interference_management",
       S::kContextualForgettingInterferenceManagement},
      {"token_efficient_kv_cache_management",
       S::kTokenEfficientKvCacheManagement},
      {"streamingllm", S::kTokenEfficientKvCacheManagement},
      {"reflection_metacognitive_buffering",
       S::kReflectionMetacognitiveBuffering},
      {"self_correcting_fact_graph", S::kSelfCorrectingFactGraph},
      {"self_rag", S::kSelfCorrectingFactGraph},
      {"slow_fast_memory_architecture", S::kSlowFastMemoryArchitecture},
      {"context_forcing", S::kSlowFastMemoryArchitecture},
      {"heat_based_tiered_migration", S::kHeatBasedTieredMigration},
      {"memoryos", S::kHeatBasedTieredMigration},
      {"a_mem", S::kHeatBasedTieredMigration},
      {"context_quarantine_isolated_scratchpads",
       S::kContextQuarantineIsolatedScratchpads},
      {"mcp_active_metadata", S::kMcpActiveMetadata},
      {"mcp", S::kMcpActiveMetadata},
  };
  if (auto it = kMap.find(normalized); it != kMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown memory strategy: ", strategy_name));
}

absl::string_view ConversationConfig::MemoryStrategyToString(
    MemoryStrategy strategy) {
  switch (strategy) {
    case MemoryStrategy::kHardResetReplayWindow:
      return "hard_reset_replay_window";
    case MemoryStrategy::kSummarizeProtectedTail:
      return "summarize_protected_tail";
    case MemoryStrategy::kVirtualMemoryPaging:
      return "virtual_memory_paging";
    case MemoryStrategy::kFactMemoryExtractionUpdate:
      return "fact_memory_extraction_update";
    case MemoryStrategy::kSemanticCompressionConsolidationAdaptiveRetrieval:
      return "semantic_compression_consolidation_adaptive_retrieval";
    case MemoryStrategy::kLearnedCompressionPolicy:
      return "learned_compression_policy";
    case MemoryStrategy::kIncrementalHierarchicalAggregation:
      return "incremental_hierarchical_aggregation";
    case MemoryStrategy::kActiveRecallSurpriseUpdate:
      return "active_recall_surprise_update";
    case MemoryStrategy::kContextualForgettingInterferenceManagement:
      return "contextual_forgetting_interference_management";
    case MemoryStrategy::kTokenEfficientKvCacheManagement:
      return "token_efficient_kv_cache_management";
    case MemoryStrategy::kReflectionMetacognitiveBuffering:
      return "reflection_metacognitive_buffering";
    case MemoryStrategy::kSelfCorrectingFactGraph:
      return "self_correcting_fact_graph";
    case MemoryStrategy::kSlowFastMemoryArchitecture:
      return "slow_fast_memory_architecture";
    case MemoryStrategy::kHeatBasedTieredMigration:
      return "heat_based_tiered_migration";
    case MemoryStrategy::kContextQuarantineIsolatedScratchpads:
      return "context_quarantine_isolated_scratchpads";
    case MemoryStrategy::kMcpActiveMetadata:
      return "mcp_active_metadata";
  }
  return "hard_reset_replay_window";
}

absl::StatusOr<ConversationConfig::SafeBoundary>
ConversationConfig::SafeBoundaryFromString(
    absl::string_view safe_boundary_name) {
  const std::string normalized = NormalizeStrategyName(safe_boundary_name);
  if (normalized == "tool_result") {
    return SafeBoundary::kToolResult;
  }
  if (normalized == "turn_boundary") {
    return SafeBoundary::kTurnBoundary;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown safe boundary: ", safe_boundary_name));
}

absl::string_view ConversationConfig::SafeBoundaryToString(
    SafeBoundary safe_boundary) {
  switch (safe_boundary) {
    case SafeBoundary::kTurnBoundary:
      return "turn_boundary";
    case SafeBoundary::kToolResult:
      return "tool_result";
  }
  return "tool_result";
}

absl::StatusOr<ConversationConfig::RuntimeMemoryPolicy>
ConversationConfig::ParseMemoryPolicyYaml(absl::string_view yaml_text) {
  ASSIGN_OR_RETURN(auto kv, ParseConstrainedYaml(yaml_text));
  RuntimeMemoryPolicy policy;

  ASSIGN_OR_RETURN(std::optional<std::string> profile_id,
                   GetYamlValue(kv, {"profile_id", "profile.id"}));
  if (profile_id.has_value()) {
    policy.profile_id = profile_id.value();
  }
  ASSIGN_OR_RETURN(std::optional<std::string> version,
                   GetYamlValue(kv, {"version", "profile.version"}));
  if (version.has_value()) {
    policy.version = version.value();
  }
  ASSIGN_OR_RETURN(std::optional<std::string> compatibility,
                   GetYamlValue(kv, {"compatibility",
                                     "profile.compatibility",
                                     "schema_compatibility"}));
  if (compatibility.has_value()) {
    policy.compatibility = compatibility.value();
  }

  ASSIGN_OR_RETURN(std::optional<std::string> strategy_text,
                   GetYamlValue(kv, {"strategy", "memory.strategy"}));
  if (!strategy_text.has_value()) {
    return absl::InvalidArgumentError("Missing required YAML key: strategy");
  }
  ASSIGN_OR_RETURN(policy.strategy, MemoryStrategyFromString(*strategy_text));

  ASSIGN_OR_RETURN(std::optional<std::string> enabled_text,
                   GetYamlValue(kv, {"context_shift_enabled",
                                     "context_shift.enabled"}));
  if (enabled_text.has_value()) {
    bool enabled = false;
    if (!TryParseBool(*enabled_text, &enabled)) {
      return absl::InvalidArgumentError(
          "context_shift_enabled must be a boolean");
    }
    policy.context_shift_enabled = enabled;
  }

  ASSIGN_OR_RETURN(std::optional<std::string> trigger_text,
                   GetYamlValue(kv, {"context_shift_trigger_ratio",
                                     "context_shift.trigger_ratio"}));
  if (trigger_text.has_value() &&
      !absl::SimpleAtof(*trigger_text, &policy.context_shift_trigger_ratio)) {
    return absl::InvalidArgumentError(
        "context_shift_trigger_ratio must be float");
  }

  ASSIGN_OR_RETURN(std::optional<std::string> retain_text,
                   GetYamlValue(kv, {"context_shift_retain_recent_messages",
                                     "context_shift.retain_recent_messages"}));
  if (retain_text.has_value() &&
      !absl::SimpleAtoi(*retain_text,
                        &policy.context_shift_retain_recent_messages)) {
    return absl::InvalidArgumentError(
        "context_shift_retain_recent_messages must be int");
  }

  ASSIGN_OR_RETURN(std::optional<std::string> target_text,
                   GetYamlValue(kv, {"context_shift_target_ratio",
                                     "context_shift.target_ratio"}));
  if (target_text.has_value() &&
      !absl::SimpleAtof(*target_text, &policy.context_shift_target_ratio)) {
    return absl::InvalidArgumentError(
        "context_shift_target_ratio must be float");
  }

  ASSIGN_OR_RETURN(std::optional<std::string> reset_text,
                   GetYamlValue(kv, {"context_shift_reset_on_exhaustion",
                                     "context_shift.reset_on_exhaustion"}));
  if (reset_text.has_value()) {
    bool reset = false;
    if (!TryParseBool(*reset_text, &reset)) {
      return absl::InvalidArgumentError(
          "context_shift_reset_on_exhaustion must be a boolean");
    }
    policy.context_shift_reset_on_exhaustion = reset;
  }

  ASSIGN_OR_RETURN(std::optional<std::string> shift_strategy_text,
                   GetYamlValue(kv, {"context_shift_strategy",
                                     "context_shift.shift_strategy"}));
  if (shift_strategy_text.has_value()) {
    const std::string normalized = NormalizeStrategyName(*shift_strategy_text);
    if (normalized == "replay_recent") {
      policy.context_shift_strategy = ContextShiftStrategy::kReplayRecent;
    } else if (normalized == "drop_all_but_system") {
      policy.context_shift_strategy = ContextShiftStrategy::kDropAllButSystem;
    } else {
      return absl::InvalidArgumentError(
          "context_shift_strategy must be replay_recent or drop_all_but_system");
    }
  }

  ASSIGN_OR_RETURN(std::optional<std::string> allow_runtime_tuning_text,
                   GetYamlValue(kv, {"allow_runtime_tuning",
                                     "overrides.allow_runtime_tuning"}));
  if (allow_runtime_tuning_text.has_value()) {
    bool allow = true;
    if (!TryParseBool(*allow_runtime_tuning_text, &allow)) {
      return absl::InvalidArgumentError(
          "allow_runtime_tuning must be a boolean");
    }
    policy.allow_runtime_tuning = allow;
  }

  ASSIGN_OR_RETURN(std::optional<std::string> safe_boundary_text,
                   GetYamlValue(kv, {"safe_boundary", "overrides.safe_boundary"}));
  if (safe_boundary_text.has_value()) {
    ASSIGN_OR_RETURN(policy.safe_boundary,
                     SafeBoundaryFromString(*safe_boundary_text));
  }

  ASSIGN_OR_RETURN(std::optional<std::string> shadow_strategy_text,
                   GetYamlValue(kv, {"shadow_strategy",
                                     "shadow.strategy"}));
  if (shadow_strategy_text.has_value()) {
    ASSIGN_OR_RETURN(policy.shadow_strategy,
                     MemoryStrategyFromString(*shadow_strategy_text));
  }

  ASSIGN_OR_RETURN(std::optional<std::string> emit_transition_note_text,
                   GetYamlValue(kv, {"emit_transition_note",
                                     "overrides.emit_transition_note"}));
  if (emit_transition_note_text.has_value()) {
    bool emit = true;
    if (!TryParseBool(*emit_transition_note_text, &emit)) {
      return absl::InvalidArgumentError(
          "emit_transition_note must be a boolean");
    }
    policy.emit_transition_note = emit;
  }

  RETURN_IF_ERROR(ValidateRuntimeMemoryPolicy(policy));
  return policy;
}

absl::StatusOr<ConversationConfig::RuntimeMemoryPolicy>
ConversationConfig::LoadMemoryPolicyYamlFile(absl::string_view yaml_file_path) {
  std::ifstream input{std::string(yaml_file_path)};
  if (!input.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open yaml file: ", yaml_file_path));
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  return ParseMemoryPolicyYaml(buffer.str());
}

absl::StatusOr<ConversationConfig> ConversationConfig::CreateInternal(
    const Engine& engine, const SessionConfig& session_config,
    std::optional<Preface> preface,
    std::optional<PromptTemplate> overwrite_prompt_template,
    std::optional<DataProcessorConfig> overwrite_processor_config,
    bool enable_constrained_decoding, bool prefill_preface_on_init,
    std::optional<ConstraintProviderConfig> constraint_provider_config,
    std::optional<std::vector<Channel>> overwrite_channels,
    bool filter_channel_content_from_kv_cache, bool context_shift_enabled,
    float context_shift_trigger_ratio,
    int context_shift_retain_recent_messages,
    float context_shift_target_ratio,
    bool context_shift_reset_on_exhaustion,
    ContextShiftStrategy context_shift_strategy,
    MemoryStrategy memory_strategy, bool allow_runtime_tuning,
    bool emit_transition_note, PolicyApplyBoundary policy_apply_boundary,
    bool prefetch_enabled, bool prefetch_shadow_mode, float prefetch_ratio) {
  if (preface.has_value() && !std::holds_alternative<JsonPreface>(*preface)) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }
  if (context_shift_trigger_ratio <= 0.0f ||
      context_shift_trigger_ratio > 1.0f) {
    return absl::InvalidArgumentError(
        "context_shift_trigger_ratio must be in (0, 1].");
  }
  if (context_shift_retain_recent_messages < 0) {
    return absl::InvalidArgumentError(
        "context_shift_retain_recent_messages must be >= 0.");
  }
  if (context_shift_target_ratio <= 0.0f ||
      context_shift_target_ratio > 1.0f) {
    return absl::InvalidArgumentError(
        "context_shift_target_ratio must be in (0, 1].");
  }
  if (context_shift_target_ratio > context_shift_trigger_ratio) {
    return absl::InvalidArgumentError(
        "context_shift_target_ratio must be <= context_shift_trigger_ratio.");
  }
  if (prefetch_ratio < 0.0f || prefetch_ratio > 1.0f) {
    return absl::InvalidArgumentError("prefetch_ratio must be in [0, 1].");
  }
  switch (context_shift_strategy) {
    case ContextShiftStrategy::kReplayRecent:
    case ContextShiftStrategy::kDropAllButSystem:
      break;
  }
  switch (memory_strategy) {
    case MemoryStrategy::kHardResetReplayWindow:
    case MemoryStrategy::kSummarizeProtectedTail:
    case MemoryStrategy::kVirtualMemoryPaging:
    case MemoryStrategy::kFactMemoryExtractionUpdate:
    case MemoryStrategy::kSemanticCompressionConsolidationAdaptiveRetrieval:
    case MemoryStrategy::kLearnedCompressionPolicy:
    case MemoryStrategy::kIncrementalHierarchicalAggregation:
    case MemoryStrategy::kActiveRecallSurpriseUpdate:
    case MemoryStrategy::kContextualForgettingInterferenceManagement:
    case MemoryStrategy::kTokenEfficientKvCacheManagement:
    case MemoryStrategy::kReflectionMetacognitiveBuffering:
    case MemoryStrategy::kSelfCorrectingFactGraph:
    case MemoryStrategy::kSlowFastMemoryArchitecture:
    case MemoryStrategy::kHeatBasedTieredMigration:
    case MemoryStrategy::kContextQuarantineIsolatedScratchpads:
    case MemoryStrategy::kMcpActiveMetadata:
      break;
  }
  if (context_shift_enabled && !prefill_preface_on_init &&
      preface.has_value() && !IsEmptyPreface(*preface)) {
    return absl::InvalidArgumentError(
        "Context shift with non-empty preface requires "
        "prefill_preface_on_init=true.");
  }

  SessionConfig session_config_copy = session_config;
  session_config_copy.SetApplyPromptTemplateInSession(false);
  RETURN_IF_ERROR(
      session_config_copy.MaybeUpdateAndValidate(engine.GetEngineSettings()));

  auto metadata = engine.GetEngineSettings().GetLlmMetadata();
  PromptTemplate prompt_template("");
  if (overwrite_prompt_template.has_value()) {
    prompt_template = *overwrite_prompt_template;
  } else if (metadata.has_value()) {
    if (metadata->has_jinja_prompt_template()) {
      prompt_template = PromptTemplate(metadata->jinja_prompt_template());
    } else if (metadata->has_prompt_templates()) {
      ASSIGN_OR_RETURN(
          std::string jinja_source,
          GetDefaultJinjaPromptTemplate(metadata->prompt_templates(),
                                        metadata->llm_model_type()));
      prompt_template = PromptTemplate(jinja_source);
    } else {
      return absl::InvalidArgumentError(
          "Failed to select jinja prompt template from llm metadata.");
    }
  } else {
    return absl::InvalidArgumentError(
        "Failed to select jinja prompt template. No llm metadata provided.");
  }

  std::vector<Channel> channels;
  if (overwrite_channels.has_value()) {
    channels = *std::move(overwrite_channels);
  } else if (metadata.has_value()) {
    for (const auto& channel : metadata->channels()) {
      channels.push_back(
          litert::lm::Channel{.channel_name = channel.channel_name(),
                              .start = channel.start(),
                              .end = channel.end()});
    }
  }

  for (const auto& channel : channels) {
    if (channel.channel_name.empty()) {
      return absl::InvalidArgumentError(
          "Custom channel must have a non-empty channel_name.");
    }
  }

  DataProcessorConfig processor_config;
  if (overwrite_processor_config.has_value()) {
    // Use the overwrite processor config if provided.
    processor_config = *overwrite_processor_config;
  } else {
    // Build the processor config from the model metadata.
    ASSIGN_OR_RETURN(processor_config,
                     CreateDataProcessorConfigFromLlmModelType(
                         session_config_copy.GetLlmModelType()));
  }

  return ConversationConfig(
      session_config_copy, preface.value_or(JsonPreface()), prompt_template,
      processor_config, enable_constrained_decoding, prefill_preface_on_init,
      std::move(constraint_provider_config), std::move(channels),
      filter_channel_content_from_kv_cache, context_shift_enabled,
      context_shift_trigger_ratio, context_shift_retain_recent_messages,
      context_shift_target_ratio, context_shift_reset_on_exhaustion,
      context_shift_strategy, memory_strategy, allow_runtime_tuning,
      emit_transition_note, policy_apply_boundary, prefetch_enabled,
      prefetch_shadow_mode, prefetch_ratio);
}

absl::StatusOr<std::string>
Conversation::GetSingleTurnTextFromSingleTurnTemplate(
    const JsonMessage& message, const OptionalArgs& optional_args) {
  absl::MutexLock lock(history_mutex_);  // NOLINT
  ASSIGN_OR_RETURN(
      auto result,
      model_data_processor_->RenderSingleTurnTemplate(
          history_,
          config_.prefill_preface_on_init() ? JsonPreface() : preface_, message,
          prompt_template_,
          /*current_is_appending_message=*/is_appending_message_,
          /*append_message=*/optional_args.has_pending_message,
          optional_args.extra_context));
  is_appending_message_ = result.is_appending_message;
  return result.text;
}

absl::StatusOr<std::string> Conversation::GetSingleTurnTextFromFullHistory(
    const JsonMessage& json_message, const OptionalArgs& optional_args) {
  PromptTemplateInput old_tmpl_input;
  RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), old_tmpl_input));

  // Merge extra context for the message into the extra context provided in the
  // preface. Existing keys will be overwritten.
  if (optional_args.extra_context.has_value()) {
    for (const auto& [key, value] : optional_args.extra_context->items()) {
      old_tmpl_input.extra_context[key] = value;
    }
  }

  absl::MutexLock lock(history_mutex_);  // NOLINT
  for (const auto& history_msg : history_) {
    if (std::holds_alternative<nlohmann::ordered_json>(history_msg)) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(
                           std::get<nlohmann::ordered_json>(history_msg)));
      old_tmpl_input.messages.push_back(message_tmpl_input);
    } else {
      return absl::UnimplementedError("Message type is not supported yet");
    }
  }
  nlohmann::ordered_json messages =
      json_message.is_array() ? json_message
                              : nlohmann::ordered_json::array({json_message});
  if (history_.empty() && !config_.prefill_preface_on_init()) {
    PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
    for (const auto& message : messages) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(message));
      new_tmpl_input.messages.push_back(message_tmpl_input);
    }
    new_tmpl_input.add_generation_prompt = true;
    return prompt_template_.Apply(new_tmpl_input);
  }

  std::string old_string;
  if (!IsEmptyPreface(preface_) || !history_.empty()) {
    old_tmpl_input.add_generation_prompt = false;
    ASSIGN_OR_RETURN(old_string, prompt_template_.Apply(old_tmpl_input));
  }

  PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
  for (const auto& message : messages) {
    ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                     model_data_processor_->MessageToTemplateInput(message));
    new_tmpl_input.messages.push_back(message_tmpl_input);
  }
  new_tmpl_input.add_generation_prompt = true;
  ASSIGN_OR_RETURN(const std::string& new_string,
                   prompt_template_.Apply(new_tmpl_input));
  if (new_string.substr(0, old_string.size()) != old_string) {
    return absl::InternalError(absl::StrCat(
        "The new rendered template string does not start with the previous "
        "rendered template string. \nold_string: ",
        old_string, "\nnew_string: ", new_string));
  }
  return {new_string.substr(old_string.size(),
                            new_string.size() - old_string.size())};
}

absl::StatusOr<std::string> Conversation::GetSingleTurnText(
    const Message& message, const OptionalArgs& optional_args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  nlohmann::ordered_json json_message =
      std::get<nlohmann::ordered_json>(message);
  if (!prompt_template_.GetCapabilities().supports_single_turn &&
      optional_args.has_pending_message) {
    return absl::InvalidArgumentError(
        "The prompt template does not support single turn template, but "
        "has_pending_message is true. `has_pending_message` is only valid for "
        "model templates and ModelDataProcessor that supports single turn "
        "prompt rendering.");
  }
  if (prompt_template_.GetCapabilities().supports_single_turn) {
    auto single_turn_text =
        GetSingleTurnTextFromSingleTurnTemplate(json_message, optional_args);
    if (!absl::IsUnimplemented(single_turn_text.status())) {
      return single_turn_text;
    }
  }
  return GetSingleTurnTextFromFullHistory(json_message, optional_args);
}

absl::StatusOr<DecodeConfig> Conversation::CreateDecodeConfig(
    std::optional<ConstraintArg> decoding_constraint,
    std::optional<int> max_output_tokens) {
  auto decode_config = DecodeConfig::CreateDefault();
  if (max_output_tokens.has_value()) {
    decode_config.SetMaxOutputTokens(max_output_tokens.value());
  }
  if (decoding_constraint.has_value() && constraint_provider_ != nullptr) {
    ASSIGN_OR_RETURN(constraint_, constraint_provider_->CreateConstraint(
                                      std::move(decoding_constraint).value()));
  } else if (config_.constrained_decoding_enabled() && constraint_ == nullptr &&
             std::holds_alternative<JsonPreface>(preface_)) {
    // Create a constraint from the tools defined in the preface, if any.
    auto json_preface = std::get<JsonPreface>(preface_);
    if (!json_preface.tools.is_null()) {
      auto constraint =
          model_data_processor_->CreateConstraint(json_preface.tools);
      if (constraint.ok()) {
        constraint_ = std::move(constraint.value());
      } else if (!absl::IsUnimplemented(constraint.status())) {
        return constraint.status();
      }
    }
  }
  decode_config.SetConstraint(constraint_.get());
  return decode_config;
}

absl::StatusOr<std::unique_ptr<Conversation>> Conversation::Create(
    Engine& engine, const ConversationConfig& config) {
  absl::Time start_time = absl::Now();
  if (!std::holds_alternative<JsonPreface>(config.GetPreface())) {
    return absl::InvalidArgumentError("Only JsonPreface is supported for now.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                   engine.CreateSession(config.GetSessionConfig()));
  ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(config.GetProcessorConfig(), config.GetPreface(),
                               &engine.GetTokenizer(),
                               session->GetSessionConfig().GetStopTokenIds(),
                               config.constrained_decoding_enabled(),
                               config.GetPromptTemplate().GetCapabilities()));
  std::unique_ptr<ConstraintProvider> constraint_provider;
  if (config.constraint_provider_config().has_value()) {
    ASSIGN_OR_RETURN(
        constraint_provider,
        CreateConstraintProvider(
            config.constraint_provider_config().value(), engine.GetTokenizer(),
            session->GetSessionConfig().GetStopTokenIds()));
  }
  auto conversation = absl::WrapUnique(new Conversation(
      engine, std::move(session), std::move(model_data_processor),
      config.GetPreface(), config.GetPromptTemplate(), config,
      std::move(constraint_provider)));
  conversation->native_cache_capabilities_ =
      conversation->session_->GetCacheOpCapabilities();
  conversation->max_context_tokens_ =
      engine.GetEngineSettings().GetMainExecutorSettings().GetMaxNumTokens();
  {
    absl::MutexLock lock(&conversation->policy_mutex_);
    conversation->active_context_shift_policy_ =
        ContextShiftRuntimePolicy{
            .context_shift_enabled = config.context_shift_enabled(),
            .context_shift_trigger_ratio = config.context_shift_trigger_ratio(),
            .context_shift_retain_recent_messages =
                config.context_shift_retain_recent_messages(),
            .context_shift_target_ratio = config.context_shift_target_ratio(),
            .context_shift_reset_on_exhaustion =
                config.context_shift_reset_on_exhaustion(),
            .context_shift_strategy = config.context_shift_strategy()};
  }
  RETURN_IF_ERROR(conversation->PrefillPrefaceIfConfigured());
  if (config.context_shift_enabled()) {
    if (!conversation->session_->SaveCheckpoint(kContextShiftAnchorCheckpoint)
             .ok()) {
      conversation->context_shift_supported_ = false;
    }
  }
  if (config.prefetch_enabled()) {
    conversation->prefetch_execution_queue_ = std::make_unique<ExecutionQueue>();
  }

  if (engine.GetEngineSettings().IsBenchmarkEnabled()) {
    ASSIGN_OR_RETURN(BenchmarkInfo * benchmark_info,
                     conversation->GetMutableBenchmarkInfo());
    RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
        BenchmarkInfo::InitPhase::kConversation, absl::Now() - start_time));
  }

  return conversation;
}

absl::Status Conversation::PrefillPrefaceIfConfigured() {
  if (!config_.prefill_preface_on_init() || IsEmptyPreface(preface_)) {
    return absl::OkStatus();
  }
  std::string single_turn_text;
  std::vector<Message> tmp_history;
  bool fallback = !prompt_template_.GetCapabilities().supports_single_turn;
  const auto render_result = model_data_processor_->RenderSingleTurnTemplate(
      tmp_history, preface_, JsonMessage(), prompt_template_,
      /*current_is_appending_message=*/false,
      /*append_message=*/false,
      /*extra_context=*/std::nullopt);
  if (fallback || absl::IsUnimplemented(render_result.status())) {
    // Fallback to the old way of prefilling the preface.
    PromptTemplateInput tmpl_input;
    RETURN_IF_ERROR(
        FillPrefaceForPromptTemplateInput(preface_, model_data_processor_.get(),
                                          tmpl_input));
    tmpl_input.add_generation_prompt = false;
    ASSIGN_OR_RETURN(single_turn_text, prompt_template_.Apply(tmpl_input));
  } else if (render_result.ok()) {
    single_turn_text = render_result->text;
  } else {
    return render_result.status();
  }
  ASSIGN_OR_RETURN(const auto session_inputs,
                   model_data_processor_->ToInputDataVector(
                       single_turn_text, std::get<JsonPreface>(preface_).messages,
                       std::monostate()));
  if (!session_inputs.empty()) {
    RETURN_IF_ERROR(session_->RunPrefill(session_inputs));
  }
  return absl::OkStatus();
}

void Conversation::AddTaskController(
    const std::optional<std::string>& task_group_id,
    std::unique_ptr<Engine::Session::TaskController> task_controller) {
  if (task_group_id.has_value() && task_controller != nullptr) {
    absl::MutexLock lock(task_controllers_mutex_);
    task_controllers_[*task_group_id].emplace_back(std::move(task_controller));
  }
}

std::vector<Conversation::PolicyTransitionRecord>
Conversation::GetPolicyTransitionRecordsForTest() const {
  absl::MutexLock lock(&policy_mutex_);
  return policy_transition_records_;
}

int Conversation::GetTransitionNoteCountForTest() const {
  absl::MutexLock lock(&policy_mutex_);
  return transition_notes_.size();
}

ConversationConfig::ContextShiftStrategy
Conversation::GetActiveContextShiftStrategyForTest() const {
  absl::MutexLock lock(&policy_mutex_);
  return active_context_shift_policy_.context_shift_strategy;
}

int Conversation::GetQueuedPolicyUpdateCountForTest() const {
  absl::MutexLock lock(&policy_mutex_);
  return pending_policy_updates_.size();
}

Conversation::PrefetchMetrics Conversation::GetPrefetchMetricsForTest() const {
  absl::MutexLock lock(&policy_mutex_);
  return prefetch_metrics_;
}

Conversation::PrefetchPlannerStateSnapshot
Conversation::GetPrefetchPlannerStateForTest() const {
  absl::MutexLock lock(&policy_mutex_);
  return PrefetchPlannerStateSnapshot{
      .lifecycle_state = prefetch_planner_state_.lifecycle_state,
      .last_invalidation_reason =
          prefetch_planner_state_.last_invalidation_reason,
      .last_plan_history_revision =
          prefetch_planner_state_.last_plan_history_revision,
      .last_plan_policy_digest = prefetch_planner_state_.last_plan_policy_digest,
      .active_plan_token = prefetch_planner_state_.active_plan_token,
      .last_plan_source_step = prefetch_planner_state_.last_plan_source_step,
      .last_successful_install_step =
          prefetch_planner_state_.last_successful_install_step,
      .last_confidence_score = prefetch_planner_state_.last_confidence_score,
  };
}

Conversation::NativeCacheStateSnapshot
Conversation::GetNativeCacheStateForTest() const {
  absl::MutexLock lock(&policy_mutex_);
  return NativeCacheStateSnapshot{
      .attempted = native_cache_state_.attempted,
      .committed = native_cache_state_.committed,
      .fallback_to_phase_b = native_cache_state_.fallback_to_phase_b,
      .last_failure_code = native_cache_state_.last_failure_code,
  };
}

bool Conversation::WaitForPrefetchPlannerStateForTest(
    PrefetchLifecycleState desired_state, absl::Duration timeout) const {
  const absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    {
      absl::MutexLock lock(&policy_mutex_);
      if (prefetch_planner_state_.lifecycle_state == desired_state) {
        return true;
      }
    }
    absl::SleepFor(absl::Milliseconds(5));
  }
  absl::MutexLock lock(&policy_mutex_);
  return prefetch_planner_state_.lifecycle_state == desired_state;
}

absl::Status Conversation::ReplaceHistoryMessageForTest(int history_index,
                                                        const Message& replacement,
                                                        bool increment_revision) {
  absl::MutexLock lock(&history_mutex_);  // NOLINT
  if (history_index < 0 ||
      history_index >= static_cast<int>(history_.size())) {
    return absl::OutOfRangeError("history_index out of range");
  }
  history_[history_index] = replacement;
  if (increment_revision) {
    ++history_revision_;
  }
  return absl::OkStatus();
}

void Conversation::RecordPrefetchMetric(
    absl::FunctionRef<void(PrefetchMetrics&)> updater) {
  absl::MutexLock lock(&policy_mutex_);
  updater(prefetch_metrics_);
}

namespace {

Conversation::PrefetchMetrics::Outcome OutcomeForReasonCode(
    Conversation::PrefetchReasonCode reason_code) {
  using Outcome = Conversation::PrefetchMetrics::Outcome;
  using Reason = Conversation::PrefetchReasonCode;
  switch (reason_code) {
    case Reason::kPlanned:
      return Outcome::kPlanned;
    case Reason::kInstalled:
      return Outcome::kInstalled;
    case Reason::kStaleDiscarded:
    case Reason::kPolicyUpdateQueued:
    case Reason::kPolicyChanged:
    case Reason::kHistoryRevisionChanged:
    case Reason::kRetainedSliceChanged:
    case Reason::kTargetStepExceeded:
    case Reason::kSupersededPlan:
      return Outcome::kStaleDiscarded;
    case Reason::kShadowSkipped:
      return Outcome::kShadowSkipped;
    case Reason::kInstallFailed:
      return Outcome::kInstallFailed;
    case Reason::kFallback:
      return Outcome::kFallback;
    case Reason::kNone:
      return Outcome::kPlanned;
  }
  return Outcome::kPlanned;
}

std::string DetectPrefetchModelType(
    const std::optional<proto::LlmMetadata>& metadata) {
  if (!metadata.has_value() || !metadata->has_llm_model_type()) {
    return "unknown";
  }
  switch (metadata->llm_model_type().model_type_case()) {
    case proto::LlmModelType::kGenericModel:
      return "generic";
    case proto::LlmModelType::kGemma3N:
      return "gemma3n";
    case proto::LlmModelType::kFunctionGemma:
      return "function_gemma";
    case proto::LlmModelType::kGemma3:
      return "gemma3";
    case proto::LlmModelType::kQwen3:
      return "qwen3";
    case proto::LlmModelType::kQwen2P5:
      return "qwen2p5";
    case proto::LlmModelType::kGemma4:
      return "gemma4";
    case proto::LlmModelType::MODEL_TYPE_NOT_SET:
      return "unknown";
  }
  return "unknown";
}

}  // namespace

Conversation::PrefetchMetrics::Dimensions
Conversation::BuildPrefetchMetricDimensions(
    const PrefetchReplayPack* pack, PrefetchReasonCode reason_code,
    std::optional<ConversationConfig::SafeBoundary> boundary) const {
  PrefetchMetrics::Dimensions dimensions;
  const ConversationConfig::RuntimeMemoryPolicy active_policy =
      GetActiveMemoryPolicy();
  dimensions.profile_id = active_policy.profile_id.value_or("");
  dimensions.strategy =
      std::string(ConversationConfig::MemoryStrategyToString(active_policy.strategy));
  dimensions.builder_id =
      pack == nullptr ? ""
                      : std::string(PrefetchBuilderIdToString(pack->builder_id));
  dimensions.boundary =
      !boundary.has_value()
          ? ""
          : std::string(ConversationConfig::SafeBoundaryToString(*boundary));
  dimensions.model_type =
      DetectPrefetchModelType(engine_.GetEngineSettings().GetLlmMetadata());
  dimensions.reason_code = std::string(PrefetchReasonCodeToString(reason_code));
  return dimensions;
}

absl::string_view Conversation::PrefetchReasonCodeToString(
    PrefetchReasonCode reason_code) {
  switch (reason_code) {
    case PrefetchReasonCode::kNone:
      return "none";
    case PrefetchReasonCode::kPlanned:
      return "planned";
    case PrefetchReasonCode::kInstalled:
      return "installed";
    case PrefetchReasonCode::kStaleDiscarded:
      return "stale_discarded";
    case PrefetchReasonCode::kShadowSkipped:
      return "shadow_skipped";
    case PrefetchReasonCode::kInstallFailed:
      return "install_failed";
    case PrefetchReasonCode::kFallback:
      return "fallback";
    case PrefetchReasonCode::kPolicyUpdateQueued:
      return "policy_update_queued";
    case PrefetchReasonCode::kPolicyChanged:
      return "policy_changed";
    case PrefetchReasonCode::kHistoryRevisionChanged:
      return "history_revision_changed";
    case PrefetchReasonCode::kRetainedSliceChanged:
      return "retained_slice_changed";
    case PrefetchReasonCode::kTargetStepExceeded:
      return "target_step_exceeded";
    case PrefetchReasonCode::kSupersededPlan:
      return "superseded_plan";
  }
  return "planned";
}

absl::string_view Conversation::NativeCacheOpVerbToString(
    NativeCacheOpVerb op_verb) {
  switch (op_verb) {
    case NativeCacheOpVerb::kPin:
      return "Pin";
    case NativeCacheOpVerb::kEvictRange:
      return "EvictRange";
    case NativeCacheOpVerb::kRemap:
      return "Remap";
    case NativeCacheOpVerb::kCompact:
      return "Compact";
    case NativeCacheOpVerb::kSnapshotRestore:
      return "SnapshotRestore";
  }
  return "Pin";
}

absl::string_view Conversation::NativeCachePinClassToString(
    NativeCachePinClass pin_class) {
  switch (pin_class) {
    case NativeCachePinClass::kSystemAnchor:
      return "system_anchor";
    case NativeCachePinClass::kAttentionSink:
      return "attention_sink";
    case NativeCachePinClass::kProtectedTail:
      return "protected_tail";
    case NativeCachePinClass::kToolState:
      return "tool_state";
    case NativeCachePinClass::kEphemeral:
      return "ephemeral";
  }
  return "ephemeral";
}

absl::string_view Conversation::NativeCacheLogicalRoleToString(
    NativeCacheLogicalRole logical_role) {
  switch (logical_role) {
    case NativeCacheLogicalRole::kSystem:
      return "system";
    case NativeCacheLogicalRole::kUser:
      return "user";
    case NativeCacheLogicalRole::kAssistant:
      return "assistant";
    case NativeCacheLogicalRole::kTool:
      return "tool";
    case NativeCacheLogicalRole::kSummaryAnchor:
      return "summary_anchor";
    case NativeCacheLogicalRole::kScratchpad:
      return "scratchpad";
  }
  return "scratchpad";
}

absl::string_view Conversation::NativeCacheFailureCodeToString(
    NativeCacheFailureCode failure_code) {
  switch (failure_code) {
    case NativeCacheFailureCode::kUnsupportedCapability:
      return "unsupported_capability";
    case NativeCacheFailureCode::kInvalidSelector:
      return "invalid_selector";
    case NativeCacheFailureCode::kRangeConflict:
      return "range_conflict";
    case NativeCacheFailureCode::kPinnedBlockConflict:
      return "pinned_block_conflict";
    case NativeCacheFailureCode::kPositionSemanticsViolation:
      return "position_semantics_violation";
    case NativeCacheFailureCode::kSummaryArtifactMissing:
      return "summary_artifact_missing";
    case NativeCacheFailureCode::kSnapshotNotFound:
      return "snapshot_not_found";
    case NativeCacheFailureCode::kRollbackUnavailable:
      return "rollback_unavailable";
    case NativeCacheFailureCode::kInternalCacheCorruptionSuspected:
      return "internal_cache_corruption_suspected";
  }
  return "unsupported_capability";
}

void Conversation::RecordPrefetchEvent(
    const PrefetchReplayPack* pack, PrefetchReasonCode reason_code,
    std::optional<ConversationConfig::SafeBoundary> boundary,
    std::optional<PrefetchInstallOutcome> install_outcome) {
  PrefetchMetrics::Event event;
  event.outcome = OutcomeForReasonCode(reason_code);
  event.dimensions = BuildPrefetchMetricDimensions(pack, reason_code, boundary);
  event.parity_mode =
      pack == nullptr ? PrefetchParityMode::kNotApplicable : pack->parity_mode;
  event.parity_mismatch = false;
  event.scaffold_only = pack != nullptr && pack->scaffold_only;

  absl::MutexLock lock(&policy_mutex_);
  prefetch_metrics_.last_dimensions = event.dimensions;
  prefetch_metrics_.last_parity_mode = event.parity_mode;
  prefetch_metrics_.last_scaffold_only = event.scaffold_only;
  prefetch_metrics_.events.push_back(std::move(event));
}

Conversation::BoundaryEvent Conversation::DetectBoundaryEvent(
    const nlohmann::ordered_json& json_msg) const {
  auto detect_from_object = [](const nlohmann::ordered_json& obj)
      -> std::optional<BoundaryEvent> {
    if (!obj.is_object() || !obj.contains(kRoleKey) || !obj[kRoleKey].is_string()) {
      return std::nullopt;
    }
    const auto role = obj[kRoleKey].get<absl::string_view>();
    if (role == kTool) {
      return BoundaryEvent::kToolResult;
    }
    return BoundaryEvent::kTurnBoundary;
  };

  if (json_msg.is_array()) {
    bool saw_turn_boundary = false;
    for (const auto& message : json_msg) {
      auto detected = detect_from_object(message);
      if (!detected.has_value()) {
        continue;
      }
      if (*detected == BoundaryEvent::kToolResult) {
        return BoundaryEvent::kToolResult;
      }
      saw_turn_boundary = true;
    }
    return saw_turn_boundary ? BoundaryEvent::kTurnBoundary
                             : BoundaryEvent::kUnknown;
  }

  if (auto detected = detect_from_object(json_msg); detected.has_value()) {
    return *detected;
  }
  return BoundaryEvent::kUnknown;
}

absl::Status Conversation::RequestPolicyUpdate(
    const ContextShiftPolicyUpdateRequest& request, BoundaryEvent boundary) {
  ContextShiftRuntimePolicy old_policy;
  ContextShiftRuntimePolicy new_policy;
  {
    absl::MutexLock lock(&policy_mutex_);
    old_policy = active_context_shift_policy_;
  }
  absl::Status validation_status = ValidatePolicyUpdateRequest(request, old_policy);
  if (!validation_status.ok()) {
    new_policy = old_policy;
  } else {
    new_policy = ResolveEffectivePolicy(old_policy, request.runtime_override);
    absl::MutexLock lock(&policy_mutex_);
    pending_policy_updates_.push_back(PendingPolicyUpdate{
        .request = request,
        .boundary = boundary,
    });
  }
  if (!validation_status.ok()) {
    RecordPolicyTransition(PolicyTransitionRecord::Action::kRejected, boundary,
                           validation_status.message(), old_policy, new_policy);
    return validation_status;
  }
  RecordPolicyTransition(PolicyTransitionRecord::Action::kQueued, boundary,
                         request.reason, old_policy, new_policy);
  return absl::OkStatus();
}

absl::Status Conversation::MaybeApplyQueuedPolicyAtBoundary(BoundaryEvent boundary) {
  if (boundary == BoundaryEvent::kUnknown) {
    return absl::OkStatus();
  }
  const bool model_turn_active = IsModelTurnActive();

  const bool should_apply_for_boundary =
      (config_.policy_apply_boundary() ==
       ConversationConfig::PolicyApplyBoundary::kToolResult &&
       boundary == BoundaryEvent::kToolResult) ||
      (config_.policy_apply_boundary() ==
       ConversationConfig::PolicyApplyBoundary::kTurnBoundary &&
       boundary == BoundaryEvent::kTurnBoundary);
  if (!should_apply_for_boundary) {
    return absl::OkStatus();
  }

  std::vector<PendingPolicyUpdate> updates_to_apply;
  {
    absl::MutexLock lock(&policy_mutex_);
    if (model_turn_active || is_appending_message_ ||
        pending_policy_updates_.empty()) {
      return absl::OkStatus();
    }
    updates_to_apply = std::move(pending_policy_updates_);
    pending_policy_updates_.clear();
  }

  for (const auto& update : updates_to_apply) {
    ContextShiftRuntimePolicy old_policy;
    ContextShiftRuntimePolicy new_policy;
    absl::Status status = absl::OkStatus();
    bool applied = false;
    {
      absl::MutexLock lock(&policy_mutex_);
      old_policy = active_context_shift_policy_;
    }
    status = ValidatePolicyUpdateRequest(update.request, old_policy);
    if (status.ok()) {
      new_policy = ResolveEffectivePolicy(old_policy, update.request.runtime_override);
      absl::MutexLock lock(&policy_mutex_);
      active_context_shift_policy_ = new_policy;
      applied = true;
    } else {
      new_policy = old_policy;
    }

    if (!status.ok()) {
      RecordPolicyTransition(PolicyTransitionRecord::Action::kRejected, boundary,
                             status.message(), old_policy, new_policy);
      continue;
    }

    if (applied && old_policy.context_shift_enabled != new_policy.context_shift_enabled &&
        new_policy.context_shift_enabled && context_shift_supported_) {
      if (!session_->SaveCheckpoint(kContextShiftAnchorCheckpoint).ok()) {
        context_shift_supported_ = false;
      }
    }
    if (applied) {
      ConversationConfig::RuntimeMemoryPolicy runtime_policy;
      {
        absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
        runtime_policy = runtime_memory_policy_override_.value_or(
            config_.runtime_memory_policy());
      }
      runtime_policy.context_shift_enabled = new_policy.context_shift_enabled;
      runtime_policy.context_shift_trigger_ratio =
          new_policy.context_shift_trigger_ratio;
      runtime_policy.context_shift_retain_recent_messages =
          new_policy.context_shift_retain_recent_messages;
      runtime_policy.context_shift_target_ratio =
          new_policy.context_shift_target_ratio;
      runtime_policy.context_shift_reset_on_exhaustion =
          new_policy.context_shift_reset_on_exhaustion;
      runtime_policy.context_shift_strategy = new_policy.context_shift_strategy;
      runtime_policy.safe_boundary =
          config_.policy_apply_boundary() ==
                  ConversationConfig::PolicyApplyBoundary::kToolResult
              ? ConversationConfig::SafeBoundary::kToolResult
              : ConversationConfig::SafeBoundary::kTurnBoundary;
      {
        absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
        runtime_memory_policy_override_ = runtime_policy;
        pending_runtime_memory_policy_update_.reset();
        policy_transition_blocked_ = false;
      }
    }
    RecordPolicyTransition(PolicyTransitionRecord::Action::kApplied, boundary,
                           update.request.reason, old_policy, new_policy);
    MaybeEmitTransitionNote(boundary, old_policy, new_policy);
  }

  return absl::OkStatus();
}

absl::Status Conversation::ValidatePolicyUpdateRequest(
    const ContextShiftPolicyUpdateRequest& request,
    const ContextShiftRuntimePolicy& current_policy) const {
  if (request.profile_schema_version != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported policy schema version: ",
        request.profile_schema_version, ". Expected 1."));
  }
  if (request.profile_compatibility_version != 1) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported policy compatibility version: ",
                     request.profile_compatibility_version, ". Expected 1."));
  }
  if (!config_.allow_runtime_tuning()) {
    return absl::FailedPreconditionError(
        "Runtime policy tuning is disabled by profile "
        "(allow_runtime_tuning=false).");
  }
  ConversationConfig::RuntimeMemoryPolicy active_runtime_policy;
  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    active_runtime_policy =
        pending_runtime_memory_policy_update_.value_or(
            runtime_memory_policy_override_.value_or(
                config_.runtime_memory_policy()));
  }
  if (!active_runtime_policy.allow_runtime_tuning) {
    return absl::FailedPreconditionError(
        "Runtime policy tuning is disabled by active policy "
        "(allow_runtime_tuning=false).");
  }

  const auto& override = request.runtime_override;
  if (override.context_shift_trigger_ratio.has_value()) {
    const float trigger = *override.context_shift_trigger_ratio;
    if (trigger <= 0.0f || trigger > 1.0f) {
      return absl::InvalidArgumentError(
          "context_shift_trigger_ratio override must be in (0, 1].");
    }
  }
  if (override.context_shift_target_ratio.has_value()) {
    const float target = *override.context_shift_target_ratio;
    if (target <= 0.0f || target > 1.0f) {
      return absl::InvalidArgumentError(
          "context_shift_target_ratio override must be in (0, 1].");
    }
  }
  if (override.context_shift_retain_recent_messages.has_value() &&
      *override.context_shift_retain_recent_messages < 0) {
    return absl::InvalidArgumentError(
        "context_shift_retain_recent_messages override must be >= 0.");
  }
  const float effective_trigger = override.context_shift_trigger_ratio.value_or(
      current_policy.context_shift_trigger_ratio);
  const float effective_target = override.context_shift_target_ratio.value_or(
      current_policy.context_shift_target_ratio);
  if (effective_target > effective_trigger) {
    return absl::InvalidArgumentError(
        "context_shift_target_ratio override must be <= "
        "context_shift_trigger_ratio after arbitration.");
  }
  return absl::OkStatus();
}

Conversation::ContextShiftRuntimePolicy Conversation::ResolveEffectivePolicy(
    const ContextShiftRuntimePolicy& current_policy,
    const ContextShiftRuntimePolicyOverride& runtime_override) const {
  ContextShiftRuntimePolicy resolved = current_policy;

  if (runtime_override.context_shift_enabled.has_value()) {
    resolved.context_shift_enabled = *runtime_override.context_shift_enabled;
  }
  if (runtime_override.context_shift_trigger_ratio.has_value()) {
    resolved.context_shift_trigger_ratio =
        *runtime_override.context_shift_trigger_ratio;
  }
  if (runtime_override.context_shift_retain_recent_messages.has_value()) {
    resolved.context_shift_retain_recent_messages =
        *runtime_override.context_shift_retain_recent_messages;
  }
  if (runtime_override.context_shift_target_ratio.has_value()) {
    resolved.context_shift_target_ratio = *runtime_override.context_shift_target_ratio;
  }
  if (runtime_override.context_shift_reset_on_exhaustion.has_value()) {
    resolved.context_shift_reset_on_exhaustion =
        *runtime_override.context_shift_reset_on_exhaustion;
  }
  if (runtime_override.context_shift_strategy.has_value()) {
    resolved.context_shift_strategy = *runtime_override.context_shift_strategy;
  }

  // Runtime/model hard caps.
  resolved.context_shift_trigger_ratio =
      std::clamp(resolved.context_shift_trigger_ratio, 0.001f, 1.0f);
  resolved.context_shift_target_ratio =
      std::clamp(resolved.context_shift_target_ratio, 0.001f, 1.0f);
  resolved.context_shift_retain_recent_messages =
      std::max(0, resolved.context_shift_retain_recent_messages);
  if (resolved.context_shift_target_ratio > resolved.context_shift_trigger_ratio) {
    resolved.context_shift_target_ratio = resolved.context_shift_trigger_ratio;
  }
  if (max_context_tokens_ <= 0) {
    resolved.context_shift_enabled = false;
  }
  return resolved;
}

std::string Conversation::SerializePolicy(
    const ContextShiftRuntimePolicy& policy) const {
  return absl::StrCat(
      "enabled=", policy.context_shift_enabled ? "true" : "false",
      ";trigger=", policy.context_shift_trigger_ratio,
      ";retain=", policy.context_shift_retain_recent_messages,
      ";target=", policy.context_shift_target_ratio,
      ";reset=", policy.context_shift_reset_on_exhaustion ? "true" : "false",
      ";strategy=", static_cast<int>(policy.context_shift_strategy));
}

void Conversation::RecordPolicyTransition(
    PolicyTransitionRecord::Action action, BoundaryEvent boundary,
    absl::string_view reason, const ContextShiftRuntimePolicy& old_policy,
    const ContextShiftRuntimePolicy& new_policy) {
  auto boundary_to_string = [](BoundaryEvent boundary_value) -> std::string {
    switch (boundary_value) {
      case BoundaryEvent::kToolResult:
        return "tool_result";
      case BoundaryEvent::kTurnBoundary:
        return "turn_boundary";
      case BoundaryEvent::kUnknown:
        return "unknown";
    }
    return "unknown";
  };

  PolicyTransitionRecord record{
      .action = action,
      .old_policy = SerializePolicy(old_policy),
      .new_policy = SerializePolicy(new_policy),
      .boundary = boundary_to_string(boundary),
      .reason = std::string(reason),
  };
  absl::MutexLock lock(&policy_mutex_);
  policy_transition_records_.push_back(std::move(record));
}

void Conversation::MaybeEmitTransitionNote(
    BoundaryEvent boundary, const ContextShiftRuntimePolicy& old_policy,
    const ContextShiftRuntimePolicy& new_policy) {
  if (!config_.emit_transition_note()) {
    return;
  }
  const std::string old_serialized = SerializePolicy(old_policy);
  const std::string new_serialized = SerializePolicy(new_policy);
  if (old_serialized == new_serialized) {
    return;
  }

  auto boundary_to_string = [](BoundaryEvent boundary_value) -> std::string {
    switch (boundary_value) {
      case BoundaryEvent::kToolResult:
        return "tool_result";
      case BoundaryEvent::kTurnBoundary:
        return "turn_boundary";
      case BoundaryEvent::kUnknown:
        return "unknown";
    }
    return "unknown";
  };

  std::string note =
      absl::StrCat("policy_transition_note|boundary=",
                   boundary_to_string(boundary), "|old=", old_serialized,
                   "|new=", new_serialized);
  {
    absl::MutexLock lock(&policy_mutex_);
    transition_notes_.push_back(note);
  }
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    history_.push_back(nlohmann::ordered_json{
        {std::string(kRoleKey), "system"}, {"content", note}});
    ++history_revision_;
  }
}

void Conversation::MaybePlanPrefetchPack(int current_step) {
  ContextShiftRuntimePolicy policy_snapshot;
  PrefetchPlannerState planner_snapshot;
  bool has_existing_pack = false;
  uint64_t existing_plan_token = 0;
  uint64_t existing_history_revision = 0;
  size_t existing_policy_digest = 0;
  size_t existing_validity_hash = 0;
  int existing_retained_start_index = -1;
  int existing_retained_end_index_exclusive = -1;
  size_t existing_retained_history_digest = 0;
  int existing_planned_target_step = 0;
  int existing_source_checkpoint_step = 0;
  float existing_confidence_score = 0.0f;
  bool has_control_plane_policy_update = false;
  {
    absl::MutexLock lock(&policy_mutex_);
    policy_snapshot = active_context_shift_policy_;
    planner_snapshot = prefetch_planner_state_;
    prefetch_planner_state_.last_observed_step = current_step;
    has_existing_pack = pending_prefetch_pack_.has_value();
    if (has_existing_pack) {
      existing_plan_token = pending_prefetch_pack_->plan_token;
      existing_history_revision = pending_prefetch_pack_->history_revision;
      existing_policy_digest = pending_prefetch_pack_->policy_digest;
      existing_validity_hash = pending_prefetch_pack_->validity_hash;
      existing_retained_start_index = pending_prefetch_pack_->retained_start_index;
      existing_retained_end_index_exclusive =
          pending_prefetch_pack_->retained_end_index_exclusive;
      existing_retained_history_digest =
          pending_prefetch_pack_->retained_history_digest;
      existing_planned_target_step = pending_prefetch_pack_->planned_target_step;
      existing_source_checkpoint_step =
          pending_prefetch_pack_->source_checkpoint_step;
      existing_confidence_score = pending_prefetch_pack_->confidence_score;
    }
    has_control_plane_policy_update = !pending_policy_updates_.empty();
  }

  bool has_runtime_policy_update = false;
  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    has_runtime_policy_update = pending_runtime_memory_policy_update_.has_value();
  }

  auto discard_pending_plan =
      [this](PrefetchInvalidationReason reason,
             PrefetchLifecycleState lifecycle_state) {
        absl::MutexLock lock(&policy_mutex_);
        queued_prefetch_task_id_.reset();
        pending_prefetch_pack_.reset();
        prefetch_planner_state_.active_plan_token = 0;
        prefetch_planner_state_.lifecycle_state = lifecycle_state;
        prefetch_planner_state_.last_invalidation_reason = reason;
      };

  auto set_idle_reason = [this](PrefetchInvalidationReason reason) {
    absl::MutexLock lock(&policy_mutex_);
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kIdle;
    prefetch_planner_state_.last_invalidation_reason = reason;
    prefetch_planner_state_.active_plan_token = 0;
  };

  if (!config_.prefetch_enabled() || max_context_tokens_ <= 0) {
    CancelQueuedPrefetchPlan(PrefetchInvalidationReason::kPrefetchDisabled);
    {
      absl::MutexLock lock(&policy_mutex_);
      prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kIdle;
    }
    return;
  }
  if (!policy_snapshot.context_shift_enabled) {
    CancelQueuedPrefetchPlan(PrefetchInvalidationReason::kContextShiftDisabled);
    {
      absl::MutexLock lock(&policy_mutex_);
      prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kIdle;
    }
    return;
  }
  if (has_control_plane_policy_update || has_runtime_policy_update) {
    CancelQueuedPrefetchPlan(PrefetchInvalidationReason::kPolicyUpdateQueued);
    return;
  }

  size_t history_size = 0;
  uint64_t history_revision = 0;
  std::vector<Message> history_snapshot;
  {
    absl::MutexLock lock(&history_mutex_);
    history_size = history_.size();
    history_revision = history_revision_;
    history_snapshot = history_;
  }
  int retained_start_index = -1;
  int retained_end_index_exclusive = -1;
  std::vector<Message> candidate_messages;
  const ConversationConfig::RuntimeMemoryPolicy active_policy =
      GetActiveMemoryPolicy();
  const PrefetchBuilderId builder_id = SelectPrefetchBuilderId(active_policy);
  std::vector<PrefetchHistoryRange> retained_ranges = BuildRetainedHistoryRanges(
      builder_id, policy_snapshot, &retained_start_index,
      &retained_end_index_exclusive, &candidate_messages);
  std::vector<PrefetchHistoryRange> protected_ranges =
      BuildProtectedHistoryRanges(builder_id, retained_ranges);
  const bool summary_anchor_present =
      builder_id == PrefetchBuilderId::kSummarizeProtectedTail;
  const bool scaffold_only =
      builder_id == PrefetchBuilderId::kSummarizeProtectedTail ||
      builder_id == PrefetchBuilderId::kQuarantineMerge;
  const PrefetchParityMode parity_mode = SelectPrefetchParityMode(builder_id);

  const size_t policy_digest = ComputePrefetchPolicyDigest(policy_snapshot);
  const size_t validity_hash = ComputePrefetchValidityHash();
  const size_t retained_history_digest = ComputeRetainedHistoryDigest(
      retained_start_index, retained_end_index_exclusive,
      policy_snapshot.context_shift_strategy);
  const size_t range_digest = ComputePrefetchArtifactIdentityDigest(
      builder_id, retained_ranges, protected_ranges, summary_anchor_present,
      scaffold_only, parity_mode, policy_snapshot);
  const int planned_target_step =
      std::max(1, static_cast<int>(max_context_tokens_ *
                                   policy_snapshot.context_shift_target_ratio));

  if (has_existing_pack) {
    const bool retained_slice_changed =
        existing_retained_start_index >= 0 &&
        existing_retained_end_index_exclusive >= existing_retained_start_index &&
        existing_retained_history_digest != retained_history_digest;
    const bool existing_plan_still_useful =
        existing_policy_digest == policy_digest &&
        existing_validity_hash == validity_hash &&
        existing_history_revision == history_revision &&
        pending_prefetch_pack_->artifact_identity_digest == range_digest &&
        !retained_slice_changed &&
        existing_planned_target_step >= current_step;
    if (existing_plan_still_useful) {
      absl::MutexLock lock(&policy_mutex_);
      prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kReady;
      prefetch_planner_state_.last_invalidation_reason =
          PrefetchInvalidationReason::kExistingPlanStillUseful;
      prefetch_planner_state_.active_plan_token = existing_plan_token;
      prefetch_planner_state_.last_plan_history_revision =
          existing_history_revision;
      prefetch_planner_state_.last_plan_policy_digest = existing_policy_digest;
      prefetch_planner_state_.last_plan_source_step =
          existing_source_checkpoint_step;
      prefetch_planner_state_.last_confidence_score =
          existing_confidence_score;
      return;
    }

    PrefetchInvalidationReason reason =
        PrefetchInvalidationReason::kHistoryRevisionChanged;
    if (existing_policy_digest != policy_digest) {
      reason = PrefetchInvalidationReason::kPolicyChanged;
    } else if (retained_slice_changed) {
      reason = PrefetchInvalidationReason::kRetainedSliceChanged;
    } else if (existing_planned_target_step < current_step) {
      reason = PrefetchInvalidationReason::kTargetStepExceeded;
    }
    CancelQueuedPrefetchPlan(reason);
  }

  if (planner_snapshot.lifecycle_state == PrefetchLifecycleState::kPlanned ||
      planner_snapshot.lifecycle_state ==
          PrefetchLifecycleState::kComputing) {
    CancelQueuedPrefetchPlan(PrefetchInvalidationReason::kSupersededPlan);
  }

  const int prefetch_trigger_step =
      std::max(1, static_cast<int>(max_context_tokens_ * config_.prefetch_ratio()));
  if (current_step < prefetch_trigger_step) {
    set_idle_reason(PrefetchInvalidationReason::kBelowTrigger);
    return;
  }

  const int step_delta = planner_snapshot.last_observed_step > 0
                             ? current_step - planner_snapshot.last_observed_step
                             : current_step;
  const int last_successful_install_step =
      planner_snapshot.last_successful_install_step;

  std::unique_ptr<ModelDataProcessor> planning_processor;
  ASSIGN_OR_RETURN(
      planning_processor,
      CreateModelDataProcessor(config_.GetProcessorConfig(), config_.GetPreface(),
                               &engine_.GetTokenizer(),
                               session_->GetSessionConfig().GetStopTokenIds(),
                               config_.constrained_decoding_enabled(),
                               config_.GetPromptTemplate().GetCapabilities()));
  absl::Status clone_status =
      planning_processor->CloneState(*model_data_processor_);
  if (!clone_status.ok() && !absl::IsUnimplemented(clone_status)) {
    discard_pending_plan(PrefetchInvalidationReason::kInstallFailed,
                         PrefetchLifecycleState::kDiscarded);
    return;
  }

  if (config_.filter_channel_content_from_kv_cache()) {
    for (auto& message : candidate_messages) {
      message = MaybeStripChannelContentFromMessage(
          message, /*strip_channel_content=*/true);
    }
  }

  if (prefetch_execution_queue_ == nullptr) {
    prefetch_execution_queue_ = std::make_unique<ExecutionQueue>();
  }

  uint64_t plan_token = 0;
  {
    absl::MutexLock lock(&policy_mutex_);
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kPlanned;
    prefetch_planner_state_.last_invalidation_reason =
        PrefetchInvalidationReason::kNone;
    prefetch_planner_state_.last_plan_history_revision = history_revision;
    prefetch_planner_state_.last_plan_policy_digest = policy_digest;
    prefetch_planner_state_.last_plan_source_step = current_step;
    prefetch_planner_state_.active_plan_token =
        ++prefetch_planner_state_.next_plan_token;
    plan_token = prefetch_planner_state_.active_plan_token;
  }

  auto queued_task_id_or = prefetch_execution_queue_->Enqueue(
      [this, plan_token, current_step, history_size, history_revision,
       retained_start_index, retained_end_index_exclusive,
       retained_history_digest, policy_digest, validity_hash,
       planned_target_step, policy_snapshot, builder_id, retained_ranges,
       protected_ranges, summary_anchor_present, scaffold_only, parity_mode,
       range_digest, step_delta,
       last_successful_install_step,
       planning_processor = std::move(planning_processor),
       candidate_messages = std::move(candidate_messages)]() mutable {
        auto mark_discarded = [this, plan_token](PrefetchInvalidationReason reason) {
          absl::MutexLock lock(&policy_mutex_);
          if (prefetch_planner_state_.active_plan_token != plan_token) {
            return;
          }
          pending_prefetch_pack_.reset();
          queued_prefetch_task_id_.reset();
          prefetch_planner_state_.active_plan_token = 0;
          prefetch_planner_state_.lifecycle_state =
              PrefetchLifecycleState::kDiscarded;
          prefetch_planner_state_.last_invalidation_reason = reason;
        };

        {
          absl::MutexLock lock(&policy_mutex_);
          if (prefetch_planner_state_.active_plan_token != plan_token) {
            return;
          }
          queued_prefetch_task_id_.reset();
          prefetch_planner_state_.lifecycle_state =
              PrefetchLifecycleState::kComputing;
        }

        bool has_runtime_policy_update = false;
        {
          absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
          has_runtime_policy_update =
              pending_runtime_memory_policy_update_.has_value();
        }
        bool has_control_plane_policy_update = false;
        size_t current_policy_digest = 0;
        {
          absl::MutexLock lock(&policy_mutex_);
          if (prefetch_planner_state_.active_plan_token != plan_token) {
            return;
          }
          has_control_plane_policy_update = !pending_policy_updates_.empty();
          current_policy_digest =
              ComputePrefetchPolicyDigest(active_context_shift_policy_);
        }
        uint64_t current_history_revision = 0;
        {
          absl::MutexLock lock(&history_mutex_);
          current_history_revision = history_revision_;
        }
        const size_t current_hash = ComputePrefetchValidityHash();
        const bool has_retained_slice =
            retained_start_index >= 0 &&
            retained_end_index_exclusive >= retained_start_index;
        const size_t current_retained_digest =
            has_retained_slice
                ? ComputeRetainedHistoryDigest(retained_start_index,
                                               retained_end_index_exclusive,
                                               policy_snapshot.context_shift_strategy)
                : 0;
        if (has_control_plane_policy_update || has_runtime_policy_update) {
          mark_discarded(PrefetchInvalidationReason::kPolicyUpdateQueued);
          return;
        }
        if (current_policy_digest != policy_digest) {
          mark_discarded(PrefetchInvalidationReason::kPolicyChanged);
          return;
        }
        if (current_history_revision != history_revision ||
            current_hash != validity_hash) {
          mark_discarded(PrefetchInvalidationReason::kHistoryRevisionChanged);
          return;
        }
        if (has_retained_slice &&
            current_retained_digest != retained_history_digest) {
          mark_discarded(PrefetchInvalidationReason::kRetainedSliceChanged);
          return;
        }

        std::vector<InputData> precomputed_replay_inputs;
        if (!candidate_messages.empty()) {
          auto replay_inputs_or = GetInputDataVectorForMessagesWithProcessor(
              *planning_processor, /*old_messages=*/absl::Span<const Message>(),
              absl::MakeConstSpan(candidate_messages), OptionalArgs());
          if (!replay_inputs_or.ok()) {
            mark_discarded(PrefetchInvalidationReason::kInstallFailed);
            return;
          }
          precomputed_replay_inputs = std::move(*replay_inputs_or);
        }

        const float confidence_score = ComputePrefetchConfidenceScore(
            current_step, step_delta, last_successful_install_step,
            policy_snapshot);

        PrefetchReplayPack pack;
        pack.plan_token = plan_token;
        pack.source_checkpoint_step = current_step;
        pack.history_watermark = history_size;
        pack.history_revision = history_revision;
        pack.builder_id = builder_id;
        pack.retained_ranges = retained_ranges;
        pack.protected_ranges = protected_ranges;
        pack.summary_anchor_present = summary_anchor_present;
        pack.scaffold_only = scaffold_only;
        pack.parity_mode = parity_mode;
        pack.retained_start_index = retained_start_index;
        pack.retained_end_index_exclusive = retained_end_index_exclusive;
        pack.retained_history_digest = retained_history_digest;
        pack.artifact_identity_digest = range_digest;
        pack.policy_digest = policy_digest;
        pack.target_ratio = policy_snapshot.context_shift_target_ratio;
        pack.confidence_score = confidence_score;
        pack.planned_target_step = planned_target_step;
        pack.strategy = policy_snapshot.context_shift_strategy;
        pack.validity_hash = validity_hash;
        pack.replay_inputs = std::move(precomputed_replay_inputs);

        has_runtime_policy_update = false;
        {
          absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
          has_runtime_policy_update =
              pending_runtime_memory_policy_update_.has_value();
        }
        {
          absl::MutexLock lock(&policy_mutex_);
          if (prefetch_planner_state_.active_plan_token != plan_token) {
            return;
          }
          has_control_plane_policy_update = !pending_policy_updates_.empty();
          current_policy_digest =
              ComputePrefetchPolicyDigest(active_context_shift_policy_);
        }
        {
          absl::MutexLock lock(&history_mutex_);
          current_history_revision = history_revision_;
        }
        if (has_control_plane_policy_update || has_runtime_policy_update) {
          mark_discarded(PrefetchInvalidationReason::kPolicyUpdateQueued);
          return;
        }
        if (current_policy_digest != policy_digest) {
          mark_discarded(PrefetchInvalidationReason::kPolicyChanged);
          return;
        }
        if (current_history_revision != history_revision ||
            ComputePrefetchValidityHash() != validity_hash) {
          mark_discarded(PrefetchInvalidationReason::kHistoryRevisionChanged);
          return;
        }
        if (has_retained_slice &&
            ComputeRetainedHistoryDigest(retained_start_index,
                                         retained_end_index_exclusive,
                                         policy_snapshot.context_shift_strategy) !=
                retained_history_digest) {
          mark_discarded(PrefetchInvalidationReason::kRetainedSliceChanged);
          return;
        }

        {
          absl::MutexLock lock(&policy_mutex_);
          if (prefetch_planner_state_.active_plan_token != plan_token) {
            return;
          }
          pending_prefetch_pack_ = std::move(pack);
          queued_prefetch_task_id_.reset();
          prefetch_planner_state_.lifecycle_state =
              PrefetchLifecycleState::kReady;
          prefetch_planner_state_.last_plan_history_revision = history_revision;
          prefetch_planner_state_.last_plan_policy_digest = policy_digest;
          prefetch_planner_state_.last_plan_source_step = current_step;
          prefetch_planner_state_.last_confidence_score = confidence_score;
        }
        RecordPrefetchMetric(
            [](PrefetchMetrics& metrics) { metrics.planned_count++; });
      });
  if (!queued_task_id_or.ok()) {
    discard_pending_plan(PrefetchInvalidationReason::kInstallFailed,
                         PrefetchLifecycleState::kDiscarded);
    return;
  }
  {
    absl::MutexLock lock(&policy_mutex_);
    if (prefetch_planner_state_.active_plan_token == plan_token &&
        prefetch_planner_state_.lifecycle_state ==
            PrefetchLifecycleState::kPlanned) {
      queued_prefetch_task_id_ = *queued_task_id_or;
    }
  }
}

absl::StatusOr<Conversation::PrefetchInstallOutcome>
Conversation::TryInstallPrefetchPackIfValid(int target_step) {
  if (!config_.prefetch_enabled()) {
    absl::MutexLock lock(&policy_mutex_);
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kIdle;
    prefetch_planner_state_.last_invalidation_reason =
        PrefetchInvalidationReason::kPrefetchDisabled;
    prefetch_planner_state_.active_plan_token = 0;
    return PrefetchInstallOutcome::kNoPendingPack;
  }

  std::optional<PrefetchReplayPack> pack;
  ContextShiftRuntimePolicy policy_snapshot;
  bool has_control_plane_policy_update = false;
  {
    absl::MutexLock lock(&policy_mutex_);
    if (!pending_prefetch_pack_.has_value()) {
      return PrefetchInstallOutcome::kNoPendingPack;
    }
    prefetch_metrics_.install_attempt_count++;
    pack = std::move(pending_prefetch_pack_);
    pending_prefetch_pack_.reset();
    policy_snapshot = active_context_shift_policy_;
    has_control_plane_policy_update = !pending_policy_updates_.empty();
  }

  bool has_runtime_policy_update = false;
  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    has_runtime_policy_update = pending_runtime_memory_policy_update_.has_value();
  }

  const size_t current_policy_digest = ComputePrefetchPolicyDigest(policy_snapshot);
  const size_t current_hash = ComputePrefetchValidityHash();
  uint64_t current_history_revision = 0;
  {
    absl::MutexLock lock(&history_mutex_);
    current_history_revision = history_revision_;
  }
  const bool has_retained_slice = pack->retained_start_index >= 0 &&
                                  pack->retained_end_index_exclusive >=
                                      pack->retained_start_index;
  const size_t current_retained_digest =
      has_retained_slice
          ? ComputeRetainedHistoryDigest(pack->retained_start_index,
                                         pack->retained_end_index_exclusive,
                                         pack->strategy)
          : 0;

  PrefetchInvalidationReason invalidation_reason =
      PrefetchInvalidationReason::kNone;
  if (has_control_plane_policy_update || has_runtime_policy_update) {
    invalidation_reason = PrefetchInvalidationReason::kPolicyUpdateQueued;
  } else if (pack->policy_digest != current_policy_digest) {
    invalidation_reason = PrefetchInvalidationReason::kPolicyChanged;
  } else if (pack->planned_target_step < target_step ||
             pack->source_checkpoint_step < target_step) {
    invalidation_reason = PrefetchInvalidationReason::kTargetStepExceeded;
  } else if (pack->history_revision != current_history_revision ||
             pack->validity_hash != current_hash) {
    invalidation_reason = PrefetchInvalidationReason::kHistoryRevisionChanged;
  }
  if (has_retained_slice &&
      current_retained_digest != pack->retained_history_digest) {
    invalidation_reason = PrefetchInvalidationReason::kRetainedSliceChanged;
  }
  if (invalidation_reason != PrefetchInvalidationReason::kNone) {
    RecordPrefetchMetric(
        [](PrefetchMetrics& metrics) { metrics.stale_discard_count++; });
    absl::MutexLock lock(&policy_mutex_);
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kDiscarded;
    prefetch_planner_state_.last_invalidation_reason = invalidation_reason;
    prefetch_planner_state_.active_plan_token = 0;
    return PrefetchInstallOutcome::kStaleDiscarded;
  }

  if (config_.prefetch_shadow_mode() || pack->replay_inputs.empty()) {
    RecordPrefetchMetric(
        [](PrefetchMetrics& metrics) { metrics.shadow_skip_count++; });
    absl::MutexLock lock(&policy_mutex_);
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kDiscarded;
    prefetch_planner_state_.last_invalidation_reason =
        PrefetchInvalidationReason::kShadowMode;
    prefetch_planner_state_.active_plan_token = 0;
    return PrefetchInstallOutcome::kShadowSkipped;
  }

  const absl::Time install_timer_start = absl::Now();
  auto rewind_status = session_->RewindToCheckpoint(kContextShiftAnchorCheckpoint);
  if (!rewind_status.ok()) {
    RecordPrefetchMetric(
        [](PrefetchMetrics& metrics) { metrics.install_failure_count++; });
    absl::MutexLock lock(&policy_mutex_);
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kDiscarded;
    prefetch_planner_state_.last_invalidation_reason =
        PrefetchInvalidationReason::kInstallFailed;
    prefetch_planner_state_.active_plan_token = 0;
    return rewind_status;
  }
  const absl::Status prefill_status =
      IgnoreEmptyInputError(session_->RunPrefill(pack->replay_inputs));
  if (!prefill_status.ok()) {
    RecordPrefetchMetric(
        [](PrefetchMetrics& metrics) { metrics.install_failure_count++; });
    absl::MutexLock lock(&policy_mutex_);
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kDiscarded;
    prefetch_planner_state_.last_invalidation_reason =
        PrefetchInvalidationReason::kInstallFailed;
    prefetch_planner_state_.active_plan_token = 0;
    return prefill_status;
  }

  const double install_latency_ms =
      absl::ToDoubleMilliseconds(absl::Now() - install_timer_start);
  {
    absl::MutexLock lock(&policy_mutex_);
    prefetch_metrics_.install_hit_count++;
    prefetch_metrics_.install_latency_ms_total += install_latency_ms;
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kInstalled;
    prefetch_planner_state_.last_invalidation_reason =
        PrefetchInvalidationReason::kNone;
    prefetch_planner_state_.active_plan_token = 0;
    prefetch_planner_state_.last_plan_history_revision = pack->history_revision;
    prefetch_planner_state_.last_plan_policy_digest = pack->policy_digest;
    prefetch_planner_state_.last_plan_source_step = pack->source_checkpoint_step;
    prefetch_planner_state_.last_successful_install_step = target_step;
    prefetch_planner_state_.last_confidence_score = pack->confidence_score;
  }
  return PrefetchInstallOutcome::kInstalled;
}

float Conversation::ComputePrefetchConfidenceScore(
    int current_step, int step_delta, int last_successful_install_step,
    const ContextShiftRuntimePolicy& policy) const {
  if (!policy.context_shift_enabled || max_context_tokens_ <= 0) {
    return 0.0f;
  }
  const float normalized_step = std::clamp(
      static_cast<float>(current_step) / static_cast<float>(max_context_tokens_),
      0.0f, 1.0f);
  const float normalized_delta = std::clamp(
      static_cast<float>(std::max(0, step_delta)) /
          std::max(1.0f, static_cast<float>(max_context_tokens_) * 0.25f),
      0.0f, 1.0f);
  const float strategy_score =
      policy.context_shift_strategy ==
              ConversationConfig::ContextShiftStrategy::kReplayRecent
          ? 1.0f
          : 0.7f;
  const float retain_window_score = std::clamp(
      1.0f - static_cast<float>(
                 std::min(policy.context_shift_retain_recent_messages, 12)) /
                 12.0f,
      0.0f, 1.0f);
  const float install_recency_score =
      last_successful_install_step < 0
          ? 0.5f
          : std::clamp(
                1.0f - static_cast<float>(
                           std::max(0, current_step -
                                           last_successful_install_step)) /
                           static_cast<float>(max_context_tokens_),
                0.0f, 1.0f);
  return std::clamp(0.35f * normalized_step + 0.25f * normalized_delta +
                        0.15f * strategy_score + 0.15f * retain_window_score +
                        0.10f * install_recency_score,
                    0.0f, 1.0f);
}

size_t Conversation::ComputePrefetchPolicyDigest(
    const ContextShiftRuntimePolicy& policy) const {
  size_t seed = 0;
  auto hash_combine = [&seed](size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  hash_combine(std::hash<bool>{}(policy.context_shift_enabled));
  hash_combine(std::hash<float>{}(policy.context_shift_trigger_ratio));
  hash_combine(std::hash<int>{}(policy.context_shift_retain_recent_messages));
  hash_combine(std::hash<float>{}(policy.context_shift_target_ratio));
  hash_combine(std::hash<bool>{}(policy.context_shift_reset_on_exhaustion));
  hash_combine(
      std::hash<int>{}(static_cast<int>(policy.context_shift_strategy)));
  return seed;
}

Conversation::PrefetchBuilderId Conversation::SelectPrefetchBuilderId(
    const ConversationConfig::RuntimeMemoryPolicy& policy) const {
  if (policy.context_shift_strategy ==
      ConversationConfig::ContextShiftStrategy::kDropAllButSystem) {
    return PrefetchBuilderId::kDropAllButSystem;
  }
  switch (policy.strategy) {
    case ConversationConfig::MemoryStrategy::kSummarizeProtectedTail:
      return PrefetchBuilderId::kSummarizeProtectedTail;
    case ConversationConfig::MemoryStrategy::
        kContextQuarantineIsolatedScratchpads:
      return PrefetchBuilderId::kQuarantineMerge;
    case ConversationConfig::MemoryStrategy::kHardResetReplayWindow:
    case ConversationConfig::MemoryStrategy::kVirtualMemoryPaging:
    case ConversationConfig::MemoryStrategy::kFactMemoryExtractionUpdate:
    case ConversationConfig::MemoryStrategy::
        kSemanticCompressionConsolidationAdaptiveRetrieval:
    case ConversationConfig::MemoryStrategy::kLearnedCompressionPolicy:
    case ConversationConfig::MemoryStrategy::
        kIncrementalHierarchicalAggregation:
    case ConversationConfig::MemoryStrategy::kActiveRecallSurpriseUpdate:
    case ConversationConfig::MemoryStrategy::
        kContextualForgettingInterferenceManagement:
    case ConversationConfig::MemoryStrategy::kTokenEfficientKvCacheManagement:
    case ConversationConfig::MemoryStrategy::kReflectionMetacognitiveBuffering:
    case ConversationConfig::MemoryStrategy::kSelfCorrectingFactGraph:
    case ConversationConfig::MemoryStrategy::kSlowFastMemoryArchitecture:
    case ConversationConfig::MemoryStrategy::kHeatBasedTieredMigration:
    case ConversationConfig::MemoryStrategy::kMcpActiveMetadata:
      return PrefetchBuilderId::kReplayRecent;
  }
  return PrefetchBuilderId::kReplayRecent;
}

std::vector<Conversation::PrefetchHistoryRange>
Conversation::BuildRetainedHistoryRanges(
    PrefetchBuilderId builder_id,
    const ContextShiftRuntimePolicy& policy_snapshot,
    int* retained_start_index, int* retained_end_index_exclusive,
    std::vector<Message>* candidate_messages) const {
  std::vector<PrefetchHistoryRange> ranges;
  size_t history_size = 0;
  {
    absl::MutexLock lock(&history_mutex_);
    history_size = history_.size();
  }
  const int history_size_int = static_cast<int>(history_size);
  if (retained_start_index != nullptr) {
    *retained_start_index = -1;
  }
  if (retained_end_index_exclusive != nullptr) {
    *retained_end_index_exclusive = -1;
  }
  if (candidate_messages != nullptr) {
    candidate_messages->clear();
  }
  if (history_size_int <= 0) {
    return ranges;
  }
  switch (builder_id) {
    case PrefetchBuilderId::kReplayRecent: {
      const int retain_count = std::min(
          history_size_int, policy_snapshot.context_shift_retain_recent_messages);
      if (retain_count > 0) {
        ranges.push_back(PrefetchHistoryRange{
            .start_message_index = history_size_int - retain_count,
            .end_message_index_exclusive = history_size_int,
        });
      }
      break;
    }
    case PrefetchBuilderId::kDropAllButSystem:
      break;
    case PrefetchBuilderId::kSummarizeProtectedTail: {
      const int protected_tail_count =
          std::min(history_size_int, std::max(1, std::min(2, history_size_int)));
      ranges.push_back(PrefetchHistoryRange{
          .start_message_index = history_size_int - protected_tail_count,
          .end_message_index_exclusive = history_size_int,
      });
      break;
    }
    case PrefetchBuilderId::kQuarantineMerge: {
      const int retain_count = std::min(
          history_size_int,
          std::max(
              1,
              std::min(policy_snapshot.context_shift_retain_recent_messages, 3)));
      ranges.push_back(PrefetchHistoryRange{
          .start_message_index = history_size_int - retain_count,
          .end_message_index_exclusive = history_size_int,
      });
      break;
    }
  }
  if (!ranges.empty()) {
    if (retained_start_index != nullptr) {
      *retained_start_index = ranges.front().start_message_index;
    }
    if (retained_end_index_exclusive != nullptr) {
      *retained_end_index_exclusive =
          ranges.back().end_message_index_exclusive;
    }
    if (candidate_messages != nullptr) {
      absl::MutexLock lock(&history_mutex_);
      for (const auto& range : ranges) {
        if (!IsValidHistoryRange(range) ||
            range.end_message_index_exclusive > history_.size()) {
          continue;
        }
        candidate_messages->insert(candidate_messages->end(),
                                   history_.begin() + range.start_message_index,
                                   history_.begin() +
                                       range.end_message_index_exclusive);
      }
    }
  }
  return ranges;
}

std::vector<Conversation::PrefetchHistoryRange>
Conversation::BuildProtectedHistoryRanges(
    PrefetchBuilderId builder_id,
    const std::vector<PrefetchHistoryRange>& retained_ranges) const {
  switch (builder_id) {
    case PrefetchBuilderId::kSummarizeProtectedTail:
    case PrefetchBuilderId::kQuarantineMerge:
      return retained_ranges;
    case PrefetchBuilderId::kReplayRecent:
    case PrefetchBuilderId::kDropAllButSystem:
      return {};
  }
  return {};
}

Conversation::PrefetchParityMode Conversation::SelectPrefetchParityMode(
    PrefetchBuilderId builder_id) const {
  switch (builder_id) {
    case PrefetchBuilderId::kReplayRecent:
    case PrefetchBuilderId::kDropAllButSystem:
      return PrefetchParityMode::kStrictTokenParity;
    case PrefetchBuilderId::kSummarizeProtectedTail:
    case PrefetchBuilderId::kQuarantineMerge:
      return PrefetchParityMode::kSemanticParity;
  }
  return PrefetchParityMode::kNotApplicable;
}

size_t Conversation::ComputePrefetchArtifactIdentityDigest(
    PrefetchBuilderId builder_id,
    absl::Span<const PrefetchHistoryRange> retained_ranges,
    absl::Span<const PrefetchHistoryRange> protected_ranges,
    bool summary_anchor_present, bool scaffold_only,
    PrefetchParityMode parity_mode,
    const ContextShiftRuntimePolicy& policy) const {
  size_t seed = 0;
  auto hash_combine = [&seed](size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  hash_combine(std::hash<int>{}(static_cast<int>(builder_id)));
  hash_combine(std::hash<int>{}(static_cast<int>(parity_mode)));
  hash_combine(std::hash<bool>{}(summary_anchor_present));
  hash_combine(std::hash<bool>{}(scaffold_only));
  hash_combine(ComputePrefetchPolicyDigest(policy));
  auto hash_range = [&hash_combine](const PrefetchHistoryRange& range) {
    hash_combine(std::hash<int>{}(range.start_message_index));
    hash_combine(std::hash<int>{}(range.end_message_index_exclusive));
  };
  for (const auto& range : retained_ranges) {
    hash_range(range);
  }
  hash_combine(std::hash<int>{}(-1));
  for (const auto& range : protected_ranges) {
    hash_range(range);
  }
  return seed;
}

absl::string_view Conversation::PrefetchBuilderIdToString(
    PrefetchBuilderId builder_id) {
  switch (builder_id) {
    case PrefetchBuilderId::kReplayRecent:
      return "replay_recent";
    case PrefetchBuilderId::kDropAllButSystem:
      return "drop_all_but_system";
    case PrefetchBuilderId::kSummarizeProtectedTail:
      return "summarize_protected_tail";
    case PrefetchBuilderId::kQuarantineMerge:
      return "quarantine_merge";
  }
  return "replay_recent";
}

absl::string_view Conversation::PrefetchParityModeToString(
    PrefetchParityMode parity_mode) {
  switch (parity_mode) {
    case PrefetchParityMode::kNotApplicable:
      return "not_applicable";
    case PrefetchParityMode::kStrictTokenParity:
      return "strict_token_parity";
    case PrefetchParityMode::kSemanticParity:
      return "semantic_parity";
  }
  return "not_applicable";
}

bool Conversation::IsValidHistoryRange(const PrefetchHistoryRange& range) {
  return range.start_message_index >= 0 &&
         range.end_message_index_exclusive >= range.start_message_index;
}

size_t Conversation::ComputePrefetchValidityHash() const {
  size_t history_size = 0;
  uint64_t history_revision = 0;
  size_t history_tail_digest = 0;
  {
    absl::MutexLock lock(&history_mutex_);
    history_size = history_.size();
    history_revision = history_revision_;
    if (!history_.empty() &&
        std::holds_alternative<nlohmann::ordered_json>(history_.back())) {
      history_tail_digest =
          std::hash<std::string>{}(std::get<nlohmann::ordered_json>(history_.back())
                                       .dump());
    }
  }

  ContextShiftRuntimePolicy policy_snapshot;
  {
    absl::MutexLock lock(&policy_mutex_);
    policy_snapshot = active_context_shift_policy_;
  }

  size_t seed = ComputePrefetchPolicyDigest(policy_snapshot);
  auto hash_combine = [&seed](size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  hash_combine(std::hash<int>{}(max_context_tokens_));
  hash_combine(std::hash<size_t>{}(history_size));
  hash_combine(std::hash<uint64_t>{}(history_revision));
  hash_combine(history_tail_digest);
  return seed;
}

size_t Conversation::ComputeRetainedHistoryDigest(
    int retained_start_index, int retained_end_index_exclusive,
    ConversationConfig::ContextShiftStrategy strategy) const {
  if (strategy != ConversationConfig::ContextShiftStrategy::kReplayRecent ||
      retained_start_index < 0 ||
      retained_end_index_exclusive < retained_start_index) {
    return 0;
  }

  std::vector<Message> retained_messages;
  {
    absl::MutexLock lock(&history_mutex_);
    if (retained_end_index_exclusive > history_.size()) {
      return 0;
    }
    retained_messages.assign(history_.begin() + retained_start_index,
                             history_.begin() + retained_end_index_exclusive);
  }
  if (config_.filter_channel_content_from_kv_cache()) {
    for (auto& message : retained_messages) {
      message = MaybeStripChannelContentFromMessage(
          message, /*strip_channel_content=*/true);
    }
  }

  size_t seed = 0;
  auto hash_combine = [&seed](size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  for (const auto& message : retained_messages) {
    if (std::holds_alternative<nlohmann::ordered_json>(message)) {
      hash_combine(std::hash<std::string>{}(
          std::get<nlohmann::ordered_json>(message).dump()));
    }
  }
  return seed;
}

absl::StatusOr<Message> Conversation::SendMessage(const Message& message,
                                                  OptionalArgs optional_args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  const BoundaryEvent incoming_boundary_event = DetectBoundaryEvent(json_message);
  if (optional_args.policy_update_request.has_value()) {
    RETURN_IF_ERROR(
        RequestPolicyUpdate(*optional_args.policy_update_request,
                            incoming_boundary_event));
  }
  RETURN_IF_ERROR(MaybeApplyQueuedPolicyAtBoundary(incoming_boundary_event));
  const ConversationConfig::SafeBoundary incoming_boundary =
      incoming_boundary_event == BoundaryEvent::kToolResult
          ? ConversationConfig::SafeBoundary::kToolResult
          : ConversationConfig::SafeBoundary::kTurnBoundary;
  RETURN_IF_ERROR(
      ApplyPendingRuntimeMemoryPolicyAtSafeBoundary(incoming_boundary));
  if (ContainsUserMessage(json_message)) {
    RETURN_IF_ERROR(MaybeApplyContextShift());
  }

  // Session inputs to be prefilled.
  std::vector<InputData> session_inputs;

  // If the incoming message is a user message, rewind to the checkpoint that
  // was saved before the assistant message containing channel content, and
  // prefill all subsequent messages with channel content removed.
  if (config_.filter_channel_content_from_kv_cache() &&
      session_checkpoint_supported_ && IsUserMessage(json_message)) {
    ASSIGN_OR_RETURN(std::vector<InputData> rewound_session_inputs,
                     RewindAndGetInputDataVector());
    session_inputs.insert(
        session_inputs.end(),
        std::make_move_iterator(rewound_session_inputs.begin()),
        std::make_move_iterator(rewound_session_inputs.end()));
  }

  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message, optional_args));
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    if (json_message.is_array()) {
      for (const auto& message : json_message) {
        history_.push_back(message);
      }
    } else {
      history_.push_back(json_message);
    }
    ++history_revision_;
  }

  ASSIGN_OR_RETURN(
      auto message_session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          optional_args.args.value_or(std::monostate())));
  session_inputs.insert(session_inputs.end(),
                        std::make_move_iterator(message_session_inputs.begin()),
                        std::make_move_iterator(message_session_inputs.end()));
  SetModelTurnActive(true);
  absl::Cleanup model_turn_cleanup = [this] { SetModelTurnActive(false); };
  RETURN_IF_ERROR(IgnoreEmptyInputError(session_->RunPrefill(session_inputs)));
  if (is_appending_message_) {
    return JsonMessage();
  } else {
    if (config_.filter_channel_content_from_kv_cache() &&
        session_checkpoint_supported_ &&
        !checkpoint_message_index_.has_value()) {
      // Before running decode, save a checkpoint for channel content
      // filtering.
      if (!session_->SaveCheckpoint(kChannelContentCheckpoint).ok()) {
        session_checkpoint_supported_ = false;
      }
    }

    ASSIGN_OR_RETURN(
        auto decode_config,
        CreateDecodeConfig(std::move(optional_args.decoding_constraint),
                           optional_args.max_output_tokens));
    ASSIGN_OR_RETURN(Responses responses, session_->RunDecode(decode_config));

    // Extract channel content from the responses. Modifies responses in place.
    ASSIGN_OR_RETURN(auto channel_content,
                     ExtractChannelContent(config_.GetChannels(), responses));

    // Convert responses to a message.
    ASSIGN_OR_RETURN(
        Message assistant_message,
        model_data_processor_->ToMessage(
            responses, optional_args.args.value_or(std::monostate())));

    // Insert channel content into the message.
    InsertChannelContentIntoMessage(channel_content, assistant_message);

    {
      absl::MutexLock lock(history_mutex_);  // NOLINT
      // Push assistant message onto history.
      history_.push_back(assistant_message);
      ++history_revision_;

      // If the assistant message contains channel content, set the checkpoint
      // message index to the current message index. This indicates the session
      // should be rewound to this message and prefilled again when the next
      // user message is sent to the model. The session checkpoint itself was
      // already saved right before the model output was decoded.
      if (config_.filter_channel_content_from_kv_cache() &&
          session_checkpoint_supported_ &&
          !checkpoint_message_index_.has_value() &&
          std::holds_alternative<nlohmann::ordered_json>(assistant_message) &&
          std::get<nlohmann::ordered_json>(assistant_message)
              .contains(kChannelsKey)) {
        checkpoint_message_index_ = history_.size() - 1;
      }
    }
    SetModelTurnActive(false);
    RETURN_IF_ERROR(
        MaybeApplyQueuedPolicyAtBoundary(BoundaryEvent::kTurnBoundary));

    std::move(model_turn_cleanup).Cancel();
    RETURN_IF_ERROR(ApplyPendingRuntimeMemoryPolicyAtSafeBoundary(
        ConversationConfig::SafeBoundary::kTurnBoundary));
    MaybeSchedulePrefetchPlanAfterBoundary();

    return assistant_message;
  }
}

absl::Status Conversation::SendMessageAsync(
    const Message& message,
    absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
    OptionalArgs optional_args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  const BoundaryEvent incoming_boundary_event = DetectBoundaryEvent(json_message);
  if (optional_args.policy_update_request.has_value()) {
    RETURN_IF_ERROR(
        RequestPolicyUpdate(*optional_args.policy_update_request,
                            incoming_boundary_event));
  }
  RETURN_IF_ERROR(MaybeApplyQueuedPolicyAtBoundary(incoming_boundary_event));
  const ConversationConfig::SafeBoundary incoming_boundary =
      incoming_boundary_event == BoundaryEvent::kToolResult
          ? ConversationConfig::SafeBoundary::kToolResult
          : ConversationConfig::SafeBoundary::kTurnBoundary;
  RETURN_IF_ERROR(
      ApplyPendingRuntimeMemoryPolicyAtSafeBoundary(incoming_boundary));
  if (ContainsUserMessage(json_message)) {
    RETURN_IF_ERROR(MaybeApplyContextShift());
  }

  // Session inputs to be prefilled.
  std::vector<InputData> session_inputs;

  // If the message is a user message, rewind to the checkpoint after the
  // previous user message and prefill all assistant messages with channel
  // content removed.
  if (config_.filter_channel_content_from_kv_cache() &&
      session_checkpoint_supported_ && IsUserMessage(json_message)) {
    ASSIGN_OR_RETURN(std::vector<InputData> rewound_session_inputs,
                     RewindAndGetInputDataVector());
    session_inputs.insert(
        session_inputs.end(),
        std::make_move_iterator(rewound_session_inputs.begin()),
        std::make_move_iterator(rewound_session_inputs.end()));
  }

  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message, optional_args));
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    if (json_message.is_array()) {
      for (const auto& message : json_message) {
        history_.push_back(message);
      }
    } else {
      history_.push_back(json_message);
    }
    ++history_revision_;
  }

  ASSIGN_OR_RETURN(
      auto message_session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          optional_args.args.value_or(std::monostate())));
  session_inputs.insert(session_inputs.end(),
                        std::make_move_iterator(message_session_inputs.begin()),
                        std::make_move_iterator(message_session_inputs.end()));

  absl::AnyInvocable<void(Message)> complete_message_callback =
      [this](const Message& complete_message) {
        {
          absl::MutexLock lock(this->history_mutex_);  // NOLINT
          this->history_.push_back(complete_message);
          ++this->history_revision_;

          // If the assistant message contains channel content, set the
          // checkpoint message index. This indicates the session should be
          // rewound to this message and prefilled again when another user
          // message is sent to the model. The session checkpoint itself was
          // already saved right before decode.
          if (config_.filter_channel_content_from_kv_cache() &&
              session_checkpoint_supported_ &&
              !checkpoint_message_index_.has_value() &&
              std::holds_alternative<nlohmann::ordered_json>(complete_message) &&
              std::get<nlohmann::ordered_json>(complete_message)
                  .contains(kChannelsKey)) {
            checkpoint_message_index_ = history_.size() - 1;
          }
        }
        SetModelTurnActive(false);
        MaybeApplyQueuedPolicyAtBoundary(BoundaryEvent::kTurnBoundary)
            .IgnoreError();
        ApplyPendingRuntimeMemoryPolicyAtSafeBoundary(
            ConversationConfig::SafeBoundary::kTurnBoundary)
            .IgnoreError();
        MaybeSchedulePrefetchPlanAfterBoundary();
      };

  absl::AnyInvocable<void()> cancel_callback = [this]() {
    absl::MutexLock lock(this->history_mutex_);  // NOLINT
    this->history_.pop_back();
    ++this->history_revision_;
    SetModelTurnActive(false);
  };

  auto internal_callback =
      std::make_shared<absl::AnyInvocable<void(absl::StatusOr<Responses>)>>(
          CreateInternalCallback(*model_data_processor_,
                                 optional_args.args.value_or(std::monostate()),
                                 config_.GetChannels(),
                                 std::move(user_callback),
                                 std::move(cancel_callback),
                                 std::move(complete_message_callback)));

  ASSIGN_OR_RETURN(
      auto decode_config,
      CreateDecodeConfig(std::move(optional_args.decoding_constraint),
                         optional_args.max_output_tokens));
  SetModelTurnActive(true);
  absl::Cleanup model_turn_cleanup = [this] { SetModelTurnActive(false); };
  if (is_appending_message_) {
    ASSIGN_OR_RETURN(
        auto task_controller,
        session_->RunPrefillAsync(
            session_inputs, [this, callback = internal_callback](
                                absl::StatusOr<Responses> responses) mutable {
              auto status = IgnoreEmptyInputError(responses.status());
              if (!status.ok()) {
                (*callback)(responses.status());
                SetModelTurnActive(false);
                return;
              }
              if (IsEmptyInputError(responses.status()) ||
                  (responses.ok() &&
                   responses->GetTaskState() == TaskState::kDone)) {
                SetModelTurnActive(false);
              }
            }));
    AddTaskController(optional_args.task_group_id, std::move(task_controller));
    std::move(model_turn_cleanup).Cancel();
  } else {
    ASSIGN_OR_RETURN(
        auto prefill_task_controller,
        session_->RunPrefillAsync(
            session_inputs, [this, callback = internal_callback, decode_config,
                             task_group_id = optional_args.task_group_id](
                                absl::StatusOr<Responses> responses) mutable {
              // First, check if prefill returned an error. Ignore errors caused
              // by empty input, as this is a valid case for triggering decode
              // only.
              auto status = IgnoreEmptyInputError(responses.status());
              // Scenario 1: Prefill failed with an unexpected error.
              if (!status.ok()) {
                // If prefill failed, invoke the callback with the error status
                // and do not proceed to decode.
                (*callback)(responses.status());
                SetModelTurnActive(false);
              } else if (IsEmptyInputError(responses.status()) ||
                         responses->GetTaskState() == TaskState::kDone) {
                // Scenario 2: Prefill was skipped due to empty input, or
                // prefill completed successfully. In either case, we can now
                // start the decode process.

                // Before running decode, save a checkpoint for channel content
                // filtering.
                if (config_.filter_channel_content_from_kv_cache() &&
                    session_checkpoint_supported_ &&
                    !checkpoint_message_index_.has_value()) {
                  // Save checkpoint in case we need to rewind later.
                  if (!session_->SaveCheckpoint(kChannelContentCheckpoint)
                           .ok()) {
                    session_checkpoint_supported_ = false;
                  }
                }

                // Run decode.
                auto decode_task_controller = session_->RunDecodeAsync(
                    [callback](absl::StatusOr<Responses> responses) {
                      (*callback)(responses);
                    },
                    decode_config);
                // If RunDecodeAsync returns a task controller, it means the
                // decode task was scheduled successfully. Add the controller
                // to our map if a task_group_id was provided, so it can be
                // cancelled later.
                if (decode_task_controller.ok()) {
                  AddTaskController(task_group_id,
                                    std::move(*decode_task_controller));
                } else {
                  // If !decode_task_controller.ok(), it means
                  // RunDecodeAsync failed to schedule. Invoke the callback
                  // with the error status.
                  (*callback)(decode_task_controller.status());
                  SetModelTurnActive(false);
                }
              }
            }));
    AddTaskController(optional_args.task_group_id,
                      std::move(prefill_task_controller));
    std::move(model_turn_cleanup).Cancel();
  }

  return absl::OkStatus();
};

absl::StatusOr<Responses> Conversation::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    OptionalArgs optional_args) {
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> cloned_session,
                   session_->Clone());
  return cloned_session->RunTextScoring(target_text,
                                        /*store_token_lengths=*/true);
}

absl::Status Conversation::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    OptionalArgs optional_args) {
  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> cloned_session,
                   session_->CloneAsync(nullptr));
  ASSIGN_OR_RETURN(auto task_controller, cloned_session->RunTextScoringAsync(
                                             target_text, std::move(callback),
                                             /*store_token_lengths=*/true));
  AddTaskController(optional_args.task_group_id, std::move(task_controller));
  return absl::OkStatus();
}

absl::StatusOr<BenchmarkInfo> Conversation::GetBenchmarkInfo() {
  return session_->GetBenchmarkInfo();
}

absl::StatusOr<BenchmarkInfo*> Conversation::GetMutableBenchmarkInfo() {
  return session_->GetMutableBenchmarkInfo();
}

void Conversation::CancelProcess() { session_->CancelProcess(); }

void Conversation::CancelGroup(absl::string_view task_group_id) {
  absl::MutexLock lock(task_controllers_mutex_);
  if (auto it = task_controllers_.find(task_group_id);
      it != task_controllers_.end()) {
    for (auto& task_controller : it->second) {
      if (task_controller != nullptr) {
        task_controller->Cancel().IgnoreError();
      }
    }
    task_controllers_.erase(it);
  }
}

void Conversation::SetModelTurnActive(bool active) {
  absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
  model_turn_active_ = active;
}

bool Conversation::IsModelTurnActive() const {
  absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
  return model_turn_active_;
}

void Conversation::QueueRuntimeMemoryPolicyUpdate(
    const ConversationConfig::RuntimeMemoryPolicy& policy) {
  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    pending_runtime_memory_policy_update_ = policy;
    policy_transition_blocked_ = true;
  }
  CancelQueuedPrefetchPlan(PrefetchInvalidationReason::kPolicyUpdateQueued);
}

absl::Status Conversation::ApplyRuntimeMemoryPolicyNow(
    const ConversationConfig::RuntimeMemoryPolicy& policy) {
  ConversationConfig::RuntimeMemoryPolicy resolved = policy;

  // Runtime override wins over static config, but still must honor
  // model/runtime hard caps.
  resolved.context_shift_trigger_ratio =
      std::clamp(resolved.context_shift_trigger_ratio, 0.001f, 1.0f);
  resolved.context_shift_target_ratio =
      std::clamp(resolved.context_shift_target_ratio, 0.001f, 1.0f);
  resolved.context_shift_retain_recent_messages =
      std::max(0, resolved.context_shift_retain_recent_messages);
  if (resolved.context_shift_target_ratio >
      resolved.context_shift_trigger_ratio) {
    resolved.context_shift_target_ratio = resolved.context_shift_trigger_ratio;
  }
  if (max_context_tokens_ <= 0) {
    resolved.context_shift_enabled = false;
  }

  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    runtime_memory_policy_override_ = resolved;
    pending_runtime_memory_policy_update_.reset();
    policy_transition_blocked_ = false;
  }
  {
    absl::MutexLock lock(&policy_mutex_);
    active_context_shift_policy_ = ContextShiftRuntimePolicy{
        .context_shift_enabled = resolved.context_shift_enabled,
        .context_shift_trigger_ratio = resolved.context_shift_trigger_ratio,
        .context_shift_retain_recent_messages =
            resolved.context_shift_retain_recent_messages,
        .context_shift_target_ratio = resolved.context_shift_target_ratio,
        .context_shift_reset_on_exhaustion =
            resolved.context_shift_reset_on_exhaustion,
        .context_shift_strategy = resolved.context_shift_strategy};
  }
  CancelQueuedPrefetchPlan(PrefetchInvalidationReason::kPolicyChanged);

  return AnchorContextForPolicyTransition(resolved);
}

absl::Status Conversation::AnchorContextForPolicyTransition(
    const ConversationConfig::RuntimeMemoryPolicy& policy) {
  if (policy.emit_transition_note) {
    nlohmann::ordered_json transition_note = {
        {std::string(kRoleKey), "system"},
        {"content",
         absl::StrCat(
             "[internal] runtime memory policy transition: strategy=",
             ConversationConfig::MemoryStrategyToString(policy.strategy),
             ", context_shift_enabled=",
             policy.context_shift_enabled ? "true" : "false",
             ", context_shift_strategy=",
             policy.context_shift_strategy ==
                     ConversationConfig::ContextShiftStrategy::kReplayRecent
                 ? "replay_recent"
                 : "drop_all_but_system")}};
    absl::MutexLock lock(history_mutex_);  // NOLINT
    history_.push_back(std::move(transition_note));
    ++history_revision_;
  }
  if (!policy.context_shift_enabled) {
    return absl::OkStatus();
  }
  absl::Status checkpoint_status =
      session_->SaveCheckpoint(kContextShiftAnchorCheckpoint);
  if (absl::IsUnimplemented(checkpoint_status)) {
    context_shift_supported_ = false;
    return absl::OkStatus();
  }
  RETURN_IF_ERROR(checkpoint_status);
  context_shift_supported_ = true;
  return absl::OkStatus();
}

absl::Status Conversation::ApplyPendingRuntimeMemoryPolicyAtSafeBoundary(
    ConversationConfig::SafeBoundary boundary) {
  std::optional<ConversationConfig::RuntimeMemoryPolicy> pending_policy;
  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    if (!pending_runtime_memory_policy_update_.has_value()) {
      return absl::OkStatus();
    }
    if (pending_runtime_memory_policy_update_->safe_boundary != boundary) {
      return absl::OkStatus();
    }
    if (model_turn_active_ || is_appending_message_) {
      policy_transition_blocked_ = true;
      return absl::OkStatus();
    }
    pending_policy = pending_runtime_memory_policy_update_;
    pending_runtime_memory_policy_update_.reset();
    policy_transition_blocked_ = false;
  }
  return ApplyRuntimeMemoryPolicyNow(*pending_policy);
}

absl::Status Conversation::SetRuntimeMemoryPolicy(
    const ConversationConfig::RuntimeMemoryPolicy& policy) {
  RETURN_IF_ERROR(ValidateRuntimeMemoryPolicy(policy));
  RETURN_IF_ERROR(ValidateRuntimePolicyVersionCompatibility(policy));

  ConversationConfig::RuntimeMemoryPolicy active_policy;
  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    if (pending_runtime_memory_policy_update_.has_value()) {
      active_policy = *pending_runtime_memory_policy_update_;
    } else {
      active_policy =
          runtime_memory_policy_override_.value_or(config_.runtime_memory_policy());
    }
  }
  if (!active_policy.allow_runtime_tuning) {
    return absl::FailedPreconditionError(
        "Runtime policy tuning is disabled by active policy "
        "(allow_runtime_tuning=false).");
  }

  QueueRuntimeMemoryPolicyUpdate(policy);
  if (!IsModelTurnActive() && !is_appending_message_ &&
      policy.safe_boundary == ConversationConfig::SafeBoundary::kTurnBoundary) {
    return ApplyPendingRuntimeMemoryPolicyAtSafeBoundary(policy.safe_boundary);
  }
  return absl::OkStatus();
}

absl::Status Conversation::SetRuntimeMemoryPolicyFromYaml(
    absl::string_view yaml_text) {
  ASSIGN_OR_RETURN(auto policy,
                   ConversationConfig::ParseMemoryPolicyYaml(yaml_text));
  return SetRuntimeMemoryPolicy(policy);
}

absl::Status Conversation::SetRuntimeMemoryPolicyFromYamlFile(
    absl::string_view yaml_file_path) {
  ASSIGN_OR_RETURN(auto policy,
                   ConversationConfig::LoadMemoryPolicyYamlFile(yaml_file_path));
  return SetRuntimeMemoryPolicy(policy);
}

ConversationConfig::RuntimeMemoryPolicy Conversation::GetActiveMemoryPolicy()
    const {
  absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
  return runtime_memory_policy_override_.value_or(
      config_.runtime_memory_policy());
}

absl::StatusOr<std::unique_ptr<Conversation>> Conversation::Clone() {
  ASSIGN_OR_RETURN(auto session, session_->Clone());
  ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(config_.GetProcessorConfig(),
                               config_.GetPreface(), &engine_.GetTokenizer(),
                               session->GetSessionConfig().GetStopTokenIds(),
                               config_.constrained_decoding_enabled(),
                               config_.GetPromptTemplate().GetCapabilities()));
  auto status = model_data_processor->CloneState(*model_data_processor_);
  if (!status.ok() && !absl::IsUnimplemented(status)) {
    return status;
  }
  std::unique_ptr<ConstraintProvider> constraint_provider;
  if (config_.constraint_provider_config().has_value()) {
    ASSIGN_OR_RETURN(constraint_provider,
                     CreateConstraintProvider(
                         config_.constraint_provider_config().value(),
                         engine_.GetTokenizer(),
                         session->GetSessionConfig().GetStopTokenIds()));
  }
  auto new_conversation = absl::WrapUnique(new Conversation(
      engine_, std::move(session), std::move(model_data_processor),
      config_.GetPreface(), config_.GetPromptTemplate(), config_,
      std::move(constraint_provider)));
  new_conversation->is_appending_message_ = is_appending_message_;
  new_conversation->context_shift_supported_ = context_shift_supported_;
  new_conversation->max_context_tokens_ = max_context_tokens_;
  new_conversation->native_cache_capabilities_ = native_cache_capabilities_;
  new_conversation->SetModelTurnActive(IsModelTurnActive());
  {
    absl::MutexLock lock(&policy_mutex_);
    new_conversation->active_context_shift_policy_ = active_context_shift_policy_;
    new_conversation->pending_policy_updates_ = pending_policy_updates_;
    new_conversation->policy_transition_records_ = policy_transition_records_;
    new_conversation->transition_notes_ = transition_notes_;
    new_conversation->queued_prefetch_task_id_.reset();
    new_conversation->pending_prefetch_pack_.reset();
    new_conversation->prefetch_metrics_ = prefetch_metrics_;
    new_conversation->native_cache_state_ = native_cache_state_;
  }
  {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    new_conversation->history_ = history_;
    new_conversation->history_revision_ = history_revision_;
  }
  {
    absl::MutexLock lock(&memory_policy_mutex_);  // NOLINT
    new_conversation->runtime_memory_policy_override_ =
        runtime_memory_policy_override_;
  }
  return new_conversation;
}

void Conversation::MaybeSchedulePrefetchPlanAfterBoundary() {
  if (!config_.prefetch_enabled() || max_context_tokens_ <= 0 ||
      !context_shift_supported_) {
    return;
  }

  auto current_step_or = session_->GetCurrentStep();
  if (!current_step_or.ok()) {
    if (absl::IsUnimplemented(current_step_or.status())) {
      context_shift_supported_ = false;
    }
    return;
  }

  MaybePlanPrefetchPack(*current_step_or);
}

void Conversation::CancelQueuedPrefetchPlan(
    PrefetchInvalidationReason reason) {
  std::optional<int> queued_task_id;
  {
    absl::MutexLock lock(&policy_mutex_);
    queued_task_id = queued_prefetch_task_id_;
    queued_prefetch_task_id_.reset();
    pending_prefetch_pack_.reset();
    prefetch_planner_state_.active_plan_token = 0;
    prefetch_planner_state_.lifecycle_state = PrefetchLifecycleState::kDiscarded;
    prefetch_planner_state_.last_invalidation_reason = reason;
  }

  if (queued_task_id.has_value() && prefetch_execution_queue_ != nullptr) {
    prefetch_execution_queue_->Remove(*queued_task_id).IgnoreError();
  }
}

absl::StatusOr<std::string> Conversation::GetPrefillTextForMessages(
    absl::Span<const Message> old_messages,
    absl::Span<const Message> new_messages, const OptionalArgs& optional_args) {
  // Create the template context for the `old` string.
  PromptTemplateInput old_context;
  old_context.add_generation_prompt = false;

  // Fill the `old` template context with the preface.
  RETURN_IF_ERROR(FillPrefaceForPromptTemplateInput(
      preface_, model_data_processor_.get(), old_context));

  // Merge extra context for the message into the extra context provided in the
  // preface. Existing keys will be overwritten.
  if (optional_args.extra_context.has_value()) {
    for (const auto& [key, value] : optional_args.extra_context->items()) {
      old_context.extra_context[key] = value;
    }
  }

  // Add old messages to the `old` template context.
  for (const auto& message : old_messages) {
    if (std::holds_alternative<nlohmann::ordered_json>(message)) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(
                           std::get<nlohmann::ordered_json>(message)));
      old_context.messages.push_back(message_tmpl_input);
    }
  }

  // Render the `old` string.
  std::string old_string;
  ASSIGN_OR_RETURN(old_string, prompt_template_.Apply(old_context));

  // Copy the `old` template context to the `new` template context.
  PromptTemplateInput new_context = old_context;

  // Add new messages to the `new` template context.
  nlohmann::ordered_json prefill_messages = nlohmann::ordered_json::array();
  for (const auto& message : new_messages) {
    if (std::holds_alternative<nlohmann::ordered_json>(message)) {
      nlohmann::ordered_json json_msg =
          std::get<nlohmann::ordered_json>(message);
      prefill_messages.push_back(json_msg);
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(json_msg));
      new_context.messages.push_back(message_tmpl_input);
    }
  }

  // Render the `new` string.
  ASSIGN_OR_RETURN(std::string new_string, prompt_template_.Apply(new_context));

  if (old_string.length() > new_string.length()) {
    return absl::InternalError(
        absl::StrCat("The new rendered string is shorter than the previous "
                     "rendered string. \nold_string: ",
                     old_string, "\nnew_string: ", new_string));
  }

  if (new_string.substr(0, old_string.size()) != old_string) {
    return absl::InternalError(
        absl::StrCat("The new rendered string does not start with the previous "
                     "rendered string. \nold_string: ",
                     old_string, "\nnew_string: ", new_string));
  }

  return new_string.substr(old_string.length());
}

absl::StatusOr<std::vector<InputData>>
Conversation::GetInputDataVectorForMessages(
    absl::Span<const Message> old_messages,
    absl::Span<const Message> new_messages, const OptionalArgs& optional_args) {
  return GetInputDataVectorForMessagesWithProcessor(
      *model_data_processor_, old_messages, new_messages, optional_args);
}

absl::StatusOr<std::vector<InputData>>
Conversation::GetInputDataVectorForMessagesWithProcessor(
    const ModelDataProcessor& processor, absl::Span<const Message> old_messages,
    absl::Span<const Message> new_messages,
    const OptionalArgs& optional_args) const {
  PromptTemplateInput old_context;
  old_context.add_generation_prompt = false;
  RETURN_IF_ERROR(
      FillPrefaceForPromptTemplateInput(preface_, &processor, old_context));
  if (optional_args.extra_context.has_value()) {
    for (const auto& [key, value] : optional_args.extra_context->items()) {
      old_context.extra_context[key] = value;
    }
  }
  for (const auto& message : old_messages) {
    if (std::holds_alternative<nlohmann::ordered_json>(message)) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       processor.MessageToTemplateInput(
                           std::get<nlohmann::ordered_json>(message)));
      old_context.messages.push_back(message_tmpl_input);
    }
  }
  ASSIGN_OR_RETURN(const std::string old_string, prompt_template_.Apply(old_context));

  PromptTemplateInput new_context = old_context;
  nlohmann::ordered_json prefill_messages = nlohmann::ordered_json::array();
  for (const auto& message : new_messages) {
    if (std::holds_alternative<nlohmann::ordered_json>(message)) {
      nlohmann::ordered_json json_msg =
          std::get<nlohmann::ordered_json>(message);
      prefill_messages.push_back(json_msg);
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       processor.MessageToTemplateInput(json_msg));
      new_context.messages.push_back(message_tmpl_input);
    }
  }
  ASSIGN_OR_RETURN(const std::string new_string, prompt_template_.Apply(new_context));
  if (old_string.length() > new_string.length()) {
    return absl::InternalError(
        absl::StrCat("The new rendered string is shorter than the previous "
                     "rendered string. \nold_string: ",
                     old_string, "\nnew_string: ", new_string));
  }
  if (new_string.substr(0, old_string.size()) != old_string) {
    return absl::InternalError(
        absl::StrCat("The new rendered string does not start with the previous "
                     "rendered string. \nold_string: ",
                     old_string, "\nnew_string: ", new_string));
  }
  const std::string prefill_text = new_string.substr(old_string.length());

  return processor.ToInputDataVector(
      prefill_text, prefill_messages,
      optional_args.args.value_or(std::monostate()));
}

absl::StatusOr<std::vector<InputData>>
Conversation::RewindAndGetInputDataVector() {
  absl::MutexLock lock(history_mutex_);
  if (!checkpoint_message_index_.has_value()) {
    // If no rewind is needed, return early with empty InputData vector.
    return std::vector<InputData>();
  }

  // Rewind the session to the saved checkpoint.
  RETURN_IF_ERROR(session_->RewindToCheckpoint(kChannelContentCheckpoint));

  // Get the InputData vector for the messages from the checkpoint onward.
  ASSIGN_OR_RETURN(
      std::vector<InputData> input_data_vector,
      GetInputDataVectorForMessages(
          absl::MakeSpan(history_).subspan(0, *checkpoint_message_index_),
          absl::MakeSpan(history_).subspan(*checkpoint_message_index_),
          OptionalArgs()));

  // Clear the checkpoint message index.
  checkpoint_message_index_ = std::nullopt;

  return input_data_vector;
}

void Conversation::RecordNativeCacheState(
    bool attempted, bool committed, bool fallback_to_phase_b,
    std::optional<NativeCacheFailureCode> last_failure_code) {
  absl::MutexLock lock(&policy_mutex_);
  native_cache_state_.attempted = attempted;
  native_cache_state_.committed = committed;
  native_cache_state_.fallback_to_phase_b = fallback_to_phase_b;
  native_cache_state_.last_failure_code = last_failure_code;
}

bool Conversation::ShouldFallbackToPhaseB(NativeCacheFailureCode failure_code) {
  switch (failure_code) {
    case NativeCacheFailureCode::kUnsupportedCapability:
    case NativeCacheFailureCode::kRollbackUnavailable:
    case NativeCacheFailureCode::kInternalCacheCorruptionSuspected:
      return true;
    case NativeCacheFailureCode::kInvalidSelector:
    case NativeCacheFailureCode::kRangeConflict:
    case NativeCacheFailureCode::kPinnedBlockConflict:
    case NativeCacheFailureCode::kPositionSemanticsViolation:
    case NativeCacheFailureCode::kSummaryArtifactMissing:
    case NativeCacheFailureCode::kSnapshotNotFound:
      return false;
  }
  return true;
}

absl::StatusOr<bool> Conversation::TryApplyNativeContextShift(int current_step,
                                                              int target_step) {
  const int tokens_to_evict = std::max(0, current_step - target_step);
  if (tokens_to_evict <= 0) {
    RecordNativeCacheState(/*attempted=*/false, /*committed=*/false,
                           /*fallback_to_phase_b=*/false, std::nullopt);
    return false;
  }
  if (!native_cache_capabilities_.supports_kv_surgery ||
      !native_cache_capabilities_.supports_range_evict) {
    RecordNativeCacheState(/*attempted=*/false, /*committed=*/false,
                           /*fallback_to_phase_b=*/false, std::nullopt);
    return false;
  }

  Engine::Session::CacheOpGroup op_group;
  op_group.requires_rollback_guarantee = true;
  op_group.ops.push_back(Engine::Session::CacheOp{
      .verb = Engine::Session::CacheOpVerb::kEvictRange,
      .token_span = {.start_token = 0, .end_token_exclusive = tokens_to_evict},
      .pin_class = Engine::Session::CachePinClass::kEphemeral,
      .logical_role = Engine::Session::CacheLogicalRole::kScratchpad,
  });

  auto cache_result_or = session_->ExecuteCacheOpGroup(op_group);
  if (!cache_result_or.ok()) {
    const NativeCacheFailureCode failure_code =
        absl::IsUnimplemented(cache_result_or.status())
            ? NativeCacheFailureCode::kUnsupportedCapability
            : NativeCacheFailureCode::kInternalCacheCorruptionSuspected;
    RecordNativeCacheState(/*attempted=*/true, /*committed=*/false,
                           ShouldFallbackToPhaseB(failure_code), failure_code);
    return false;
  }

  const Engine::Session::CacheOpGroupResult& cache_result = *cache_result_or;
  if (cache_result.committed && !cache_result.failure.has_value()) {
    RecordNativeCacheState(/*attempted=*/true, /*committed=*/true,
                           /*fallback_to_phase_b=*/false, std::nullopt);
    return true;
  }

  NativeCacheFailureCode failure_code =
      cache_result.failure.has_value()
          ? cache_result.failure->code
          : NativeCacheFailureCode::kInternalCacheCorruptionSuspected;
  if (!cache_result.rollback_available) {
    failure_code = NativeCacheFailureCode::kRollbackUnavailable;
  }
  RecordNativeCacheState(/*attempted=*/true, /*committed=*/false,
                         ShouldFallbackToPhaseB(failure_code), failure_code);
  return false;
}

absl::Status Conversation::MaybeApplyContextShift() {
  const ConversationConfig::RuntimeMemoryPolicy policy = GetActiveMemoryPolicy();
  if (!policy.context_shift_enabled || !context_shift_supported_ ||
      max_context_tokens_ <= 0 || is_appending_message_) {
    return absl::OkStatus();
  }

  auto current_step_or = session_->GetCurrentStep();
  if (!current_step_or.ok()) {
    if (absl::IsUnimplemented(current_step_or.status())) {
      context_shift_supported_ = false;
      return absl::OkStatus();
    }
    return current_step_or.status();
  }

  const int trigger_step =
      std::max(1, static_cast<int>(max_context_tokens_ *
                                   policy.context_shift_trigger_ratio));
  if (*current_step_or < trigger_step) {
    return absl::OkStatus();
  }

  const bool use_replay_recent =
      policy.context_shift_strategy ==
      ConversationConfig::ContextShiftStrategy::kReplayRecent;
  std::vector<Message> candidate_messages;
  if (use_replay_recent) {
    absl::MutexLock lock(history_mutex_);  // NOLINT
    const int retain_count =
        std::min(static_cast<int>(history_.size()),
                 policy.context_shift_retain_recent_messages);
    if (retain_count > 0) {
      candidate_messages.assign(history_.end() - retain_count, history_.end());
    }
    if (config_.filter_channel_content_from_kv_cache()) {
      for (auto& message : candidate_messages) {
        message = MaybeStripChannelContentFromMessage(
            message, /*strip_channel_content=*/true);
      }
    }
  }

  const int target_step =
      std::max(1, static_cast<int>(max_context_tokens_ *
                                   policy.context_shift_target_ratio));
  bool attempted_prefetch_install = false;
  bool installed_prefetch = false;
  if (config_.prefetch_enabled()) {
    ASSIGN_OR_RETURN(const PrefetchInstallOutcome install_outcome,
                     TryInstallPrefetchPackIfValid(target_step));
    attempted_prefetch_install =
        install_outcome != PrefetchInstallOutcome::kNoPendingPack;
    installed_prefetch = install_outcome == PrefetchInstallOutcome::kInstalled;
  }

  int replay_count = static_cast<int>(candidate_messages.size());
  int shifted_step = *current_step_or;
  bool baseline_recompute_needed = true;
  if (installed_prefetch) {
    baseline_recompute_needed = false;
    shifted_step = target_step;
    RecordPrefetchMetric(
        [](PrefetchMetrics& metrics) { metrics.parity_check_count++; });
  }

  if (baseline_recompute_needed) {
    ASSIGN_OR_RETURN(const bool native_shift_applied,
                     TryApplyNativeContextShift(*current_step_or, target_step));
    if (native_shift_applied) {
      baseline_recompute_needed = false;
      shifted_step = target_step;
    }
  }

  if (baseline_recompute_needed) {
    const absl::Time baseline_timer_start = absl::Now();
    while (true) {
      auto rewind_status =
          session_->RewindToCheckpoint(kContextShiftAnchorCheckpoint);
      if (!rewind_status.ok()) {
        if (absl::IsUnimplemented(rewind_status)) {
          context_shift_supported_ = false;
          return absl::OkStatus();
        }
        return rewind_status;
      }

      if (use_replay_recent && replay_count > 0) {
        ASSIGN_OR_RETURN(std::vector<InputData> replay_inputs,
                         GetInputDataVectorForMessages(
                             /*old_messages=*/absl::Span<const Message>(),
                             absl::MakeSpan(candidate_messages).first(replay_count),
                             OptionalArgs()));
        RETURN_IF_ERROR(
            IgnoreEmptyInputError(session_->RunPrefill(replay_inputs)));
      }

      ASSIGN_OR_RETURN(shifted_step, session_->GetCurrentStep());
      if (shifted_step <= target_step ||
          (!use_replay_recent || replay_count == 0)) {
        break;
      }
      --replay_count;
    }

    const double baseline_latency_ms =
        absl::ToDoubleMilliseconds(absl::Now() - baseline_timer_start);
    RecordPrefetchMetric([baseline_latency_ms, attempted_prefetch_install,
                          installed_prefetch](PrefetchMetrics& metrics) {
      metrics.baseline_recompute_latency_ms_total += baseline_latency_ms;
      if (attempted_prefetch_install) {
        metrics.fallback_count++;
        if (installed_prefetch) {
          metrics.parity_mismatch_count++;
        }
      }
      metrics.parity_check_count++;
    });
  }

  if (shifted_step > target_step && replay_count == 0 &&
      policy.context_shift_reset_on_exhaustion) {
    ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> new_session,
                     engine_.CreateSession(config_.GetSessionConfig()));
    session_ = std::move(new_session);
    native_cache_capabilities_ = session_->GetCacheOpCapabilities();
    RETURN_IF_ERROR(PrefillPrefaceIfConfigured());
  }

  checkpoint_message_index_ = std::nullopt;
  if (!session_->SaveCheckpoint(kContextShiftAnchorCheckpoint).ok()) {
    context_shift_supported_ = false;
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
