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

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/prompt_template.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::InSequence;
using ::testing::Not;
using ::testing::ResultOf;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::VariantWith;

absl::string_view kTestLlmPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";

constexpr char kTestTokenizerPath[] =
    "litert_lm/runtime/components/testdata/gemma3_sentencepiece.model";

constexpr char kGemma3ToolsMultiPrefillTemplatePath[] =
    "litert_lm/runtime/components/testdata/"
    "google-gemma-3n-e2b-it-tools-multi-prefill.jinja";

constexpr char kGemma3TemplatePath[] =
    "litert_lm/runtime/components/testdata/google-gemma-3-1b-it.jinja";

constexpr absl::string_view kTestJinjaPromptTemplate = R"jinja(
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

std::string GetTestdataPath(absl::string_view file_path) {
  return absl::StrCat(::testing::SrcDir(), "/", file_path);
}

std::string ReadFile(absl::string_view path) {
  std::ifstream ifstr(std::string(path), std::ios::binary);
  std::stringstream contents;
  contents << ifstr.rdbuf();
  return contents.str();
}

class MockSession : public Engine::Session {
 public:
  MOCK_METHOD(absl::StatusOr<Responses>, GenerateContent,
              (const std::vector<InputData>& contents), (override));
  MOCK_METHOD(
      absl::Status, GenerateContentStream,
      (const std::vector<InputData>& contents,
       absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback),
      (override));
  MOCK_METHOD(
      absl::Status, GenerateContentStream,
      (const std::vector<InputData>& contents,
       absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
       const DecodeConfig& decode_config),
      (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunTextScoring,
              (const std::vector<absl::string_view>& target_text,
               bool store_token_lengths),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
              RunTextScoringAsync,
              (const std::vector<absl::string_view>& target_text,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
               bool store_token_lengths),
              (override));

  MOCK_METHOD(absl::Status, RunPrefill,
              (const std::vector<InputData>& contents), (override));
  MOCK_METHOD(
      absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
      RunPrefillAsync,
      (const std::vector<InputData>& contents,
       absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback),
      (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode, (), (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode,
              (const DecodeConfig& decode_config), (override));
  MOCK_METHOD(
      absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
      RunDecodeAsync,
      (absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback),
      (override));
  MOCK_METHOD(
      absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
      RunDecodeAsync,
      (absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
       const DecodeConfig& decode_config),
      (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, Clone, (), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, CloneAsync,
              (absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback),
              (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo>, GetBenchmarkInfo, (), (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo*>, GetMutableBenchmarkInfo, (),
              (override));
  MOCK_METHOD(void, CancelProcess, (), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (), (override));
  MOCK_METHOD(absl::Status, SaveCheckpoint, (absl::string_view label),
              (override));
  MOCK_METHOD(absl::Status, RewindToCheckpoint, (absl::string_view label),
              (override));
  MOCK_METHOD(absl::StatusOr<int>, GetCurrentStep, (), (const, override));
  MOCK_METHOD(Engine::Session::CacheOpCapabilities, GetCacheOpCapabilities, (),
              (const, override));
  MOCK_METHOD((absl::StatusOr<Engine::Session::CacheOpGroupResult>),
              ExecuteCacheOpGroup,
              (const Engine::Session::CacheOpGroup& cache_op_group), (override));
  MOCK_METHOD(const SessionConfig&, GetSessionConfig, (), (const, override));
};

class MockEngine : public Engine {
 public:
  MOCK_METHOD(const EngineSettings&, GetEngineSettings, (), (const, override));
  MOCK_METHOD(const Tokenizer&, GetTokenizer, (), (const, override));
  MOCK_METHOD(absl::StatusOr<AudioExecutorProperties>,
              GetAudioExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<VisionExecutorProperties>,
              GetVisionExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, CreateSession,
              (const SessionConfig& session_config), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (absl::Duration timeout),
              (override));
};

class MockTaskController : public Engine::Session::TaskController {
 public:
  MockTaskController() = default;
  ~MockTaskController() override = default;
  MOCK_METHOD(absl::Status, Cancel, (), (override));
};

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateTestMessageCallback(
    Message& expected_message, absl::Notification& done) {
  return [&expected_message, &done](absl::StatusOr<Message> message) mutable {
    // If the message is not ok, fail the test.
    if (!message.ok()) {
      FAIL() << "Message user_callback failed: " << message.status();
      return;
    }
    // If the message is null, the last callback is received.
    if (auto json_message = std::get_if<JsonMessage>(&message.value());
        json_message->is_null()) {
      JsonMessage& expected_json_message =
          std::get<JsonMessage>(expected_message);
      ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
      std::string expected_string = expected_json_message["content"][0]["text"];
      // The expected string should be empty after the last callback.
      EXPECT_TRUE(expected_string.empty());
      done.Notify();
      return;
    }
    // Otherwise, this is a partial response.
    if (auto json_message = std::get_if<JsonMessage>(&message.value())) {
      JsonMessage& expected_json_message =
          std::get<JsonMessage>(expected_message);
      // Compare the message text content by prefix, and update the expected
      // message to the remaining text for the next user_callback.
      ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
      ASSERT_TRUE((*json_message)["content"][0]["text"].is_string());
      std::string expected_string = expected_json_message["content"][0]["text"];
      std::string actual_string = (*json_message)["content"][0]["text"];
      EXPECT_TRUE(absl::StartsWith(expected_string, actual_string))
          << "Expected: " << expected_string << "\nActual: " << actual_string;
      expected_json_message["content"][0]["text"] =
          expected_string.substr(actual_string.size());
    }
  };
}

void ExpectAssistantMessageWithNonEmptyText(const Message& message) {
  ASSERT_TRUE(std::holds_alternative<JsonMessage>(message));
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  ASSERT_TRUE(json_message.is_object());
  ASSERT_TRUE(json_message.contains("role"));
  ASSERT_TRUE(json_message["role"].is_string());
  EXPECT_EQ(json_message["role"], "assistant");
  ASSERT_TRUE(json_message.contains("content"));
  const auto& content = json_message["content"];
  if (content.is_string()) {
    EXPECT_FALSE(content.get<std::string>().empty());
    return;
  }
  ASSERT_TRUE(content.is_array());
  ASSERT_FALSE(content.empty());
  ASSERT_TRUE(content[0].is_object());
  ASSERT_TRUE(content[0].contains("text"));
  ASSERT_TRUE(content[0]["text"].is_string());
  EXPECT_FALSE(content[0]["text"].get<std::string>().empty());
}

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateStreamingObserverCallback(
    int& partial_message_count, absl::Notification& done) {
  return [&partial_message_count, &done](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      FAIL() << "Message user_callback failed: " << message.status();
      done.Notify();
      return;
    }
    if (auto json_message = std::get_if<JsonMessage>(&message.value());
        json_message->is_null()) {
      done.Notify();
      return;
    }
    ++partial_message_count;
  };
}

absl::AnyInvocable<void(absl::StatusOr<Message>)>
CreateTestMultiMessageCallback(const std::vector<Message>& expected_messages,
                               absl::Notification& done) {
  return [&expected_messages, &done,
          current_index = 0](absl::StatusOr<Message> message) mutable {
    ASSERT_OK(message);
    ASSERT_TRUE(std::holds_alternative<JsonMessage>(message.value()));
    auto json_message = std::get<JsonMessage>(message.value());

    // If the message is null, the message stream is complete.
    if (json_message.is_null()) {
      EXPECT_TRUE(current_index == expected_messages.size())
          << "Expected " << expected_messages.size()
          << " messages but only got " << current_index;
      done.Notify();
      return;
    }

    ASSERT_LT(current_index, expected_messages.size())
        << "Received more messages than expected. Expected size: "
        << expected_messages.size();
    EXPECT_THAT(*message, testing::Eq(expected_messages[current_index]));
    ++current_index;
  };
}

TEST(ConversationConfigTest, CreateDefault) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  EXPECT_OK(Conversation::Create(*engine, config));
}

TEST(ConversationConfigTest, CreateDefaultWithOverwritePromptTemplate) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::Builder()
                                        .SetOverwritePromptTemplate(
                                            PromptTemplate("Hello world!"))
                                        .Build(*engine));
  EXPECT_EQ(config.GetPromptTemplate().GetTemplateSource(), "Hello world!");
  EXPECT_TRUE(
      config.GetSessionConfig().GetPromptTemplates().user().prefix().empty());
}

TEST(ConversationConfigTest, CreateWithBuilder) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  auto session_config = SessionConfig::CreateDefault();
  session_config.GetMutableLlmModelType().mutable_gemma3n();

  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .Build(*engine));
  EXPECT_TRUE(std::holds_alternative<JsonPreface>(config.GetPreface()));
  EXPECT_EQ(
      std::get<JsonPreface>(config.GetPreface()).messages,
      nlohmann::ordered_json(
          {{{"role", "system"}, {"content", "You are a helpful assistant."}}}));
  EXPECT_EQ(config.GetSessionConfig().GetLlmModelType().model_type_case(),
            proto::LlmModelType::kGemma3N);
  EXPECT_TRUE(
      config.GetSessionConfig().GetPromptTemplates().user().prefix().empty());
  EXPECT_OK(Conversation::Create(*engine, config));
}

TEST(ConversationConfigTest, FilterChannelContentFromKvCache) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  ASSERT_OK_AND_ASSIGN(auto config,
                       ConversationConfig::Builder()
                           .SetFilterChannelContentFromKvCache(true)
                           .Build(*engine));
  EXPECT_TRUE(config.filter_channel_content_from_kv_cache());
}

TEST(ConversationConfigTest, OverwritePromptTemplate) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetOverwritePromptTemplate(PromptTemplate("overwrite template"))
          .Build(*engine));

  EXPECT_EQ(config.GetPromptTemplate().GetTemplateSource(),
            "overwrite template");
}

TEST(ConversationConfigTest, ContextShiftConfigValidation) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  auto invalid_ratio = ConversationConfig::Builder()
                           .SetEnableContextShift(true)
                           .SetContextShiftTriggerRatio(0.0f)
                           .Build(*engine);
  EXPECT_FALSE(invalid_ratio.ok());

  auto invalid_retain = ConversationConfig::Builder()
                            .SetEnableContextShift(true)
                            .SetContextShiftRetainRecentMessages(-1)
                            .Build(*engine);
  EXPECT_FALSE(invalid_retain.ok());

  auto invalid_target = ConversationConfig::Builder()
                            .SetEnableContextShift(true)
                            .SetContextShiftTriggerRatio(0.5f)
                            .SetContextShiftTargetRatio(0.6f)
                            .Build(*engine);
  EXPECT_FALSE(invalid_target.ok());

  auto invalid_preface = ConversationConfig::Builder()
                             .SetEnableContextShift(true)
                             .SetPreface(JsonPreface{
                                 .messages = {{{"role", "system"},
                                               {"content", "hi"}}}})
                             .SetPrefillPrefaceOnInit(false)
                             .Build(*engine);
  EXPECT_FALSE(invalid_preface.ok());
}

TEST(ConversationConfigTest, ParseMemoryPolicyYamlSupportsAllStrategies) {
  struct TestCase {
    absl::string_view strategy;
    ConversationConfig::MemoryStrategy expected;
  };

  const std::vector<TestCase> test_cases = {
      {"hard_reset_replay_window",
       ConversationConfig::MemoryStrategy::kHardResetReplayWindow},
      {"summarize_protected_tail",
       ConversationConfig::MemoryStrategy::kSummarizeProtectedTail},
      {"virtual_memory_paging",
       ConversationConfig::MemoryStrategy::kVirtualMemoryPaging},
      {"fact_memory_extraction_update",
       ConversationConfig::MemoryStrategy::kFactMemoryExtractionUpdate},
      {"semantic_compression_consolidation_adaptive_retrieval",
       ConversationConfig::MemoryStrategy::
           kSemanticCompressionConsolidationAdaptiveRetrieval},
      {"learned_compression_policy",
       ConversationConfig::MemoryStrategy::kLearnedCompressionPolicy},
      {"incremental_hierarchical_aggregation",
       ConversationConfig::MemoryStrategy::kIncrementalHierarchicalAggregation},
      {"active_recall_surprise_update",
       ConversationConfig::MemoryStrategy::kActiveRecallSurpriseUpdate},
      {"contextual_forgetting_interference_management",
       ConversationConfig::MemoryStrategy::
           kContextualForgettingInterferenceManagement},
      {"token_efficient_kv_cache_management",
       ConversationConfig::MemoryStrategy::kTokenEfficientKvCacheManagement},
      {"reflection_metacognitive_buffering",
       ConversationConfig::MemoryStrategy::kReflectionMetacognitiveBuffering},
      {"self_correcting_fact_graph",
       ConversationConfig::MemoryStrategy::kSelfCorrectingFactGraph},
      {"slow_fast_memory_architecture",
       ConversationConfig::MemoryStrategy::kSlowFastMemoryArchitecture},
      {"heat_based_tiered_migration",
       ConversationConfig::MemoryStrategy::kHeatBasedTieredMigration},
      {"context_quarantine_isolated_scratchpads",
       ConversationConfig::MemoryStrategy::
           kContextQuarantineIsolatedScratchpads},
      {"mcp_active_metadata",
       ConversationConfig::MemoryStrategy::kMcpActiveMetadata},
  };

  for (const auto& tc : test_cases) {
    const std::string yaml = absl::StrCat(
        "profile_id: shared-profile\n"
        "strategy: ",
        tc.strategy,
        "\n"
        "context_shift:\n"
        "  enabled: true\n"
        "  trigger_ratio: 0.9\n"
        "  retain_recent_messages: 8\n"
        "  target_ratio: 0.8\n"
        "  reset_on_exhaustion: true\n"
        "  shift_strategy: replay_recent\n");

    ASSERT_OK_AND_ASSIGN(auto policy,
                         ConversationConfig::ParseMemoryPolicyYaml(yaml));
    EXPECT_EQ(policy.strategy, tc.expected) << "strategy=" << tc.strategy;
    EXPECT_TRUE(policy.context_shift_enabled);
    EXPECT_EQ(policy.context_shift_retain_recent_messages, 8);
    EXPECT_EQ(policy.context_shift_strategy,
              ConversationConfig::ContextShiftStrategy::kReplayRecent);
    ASSERT_TRUE(policy.profile_id.has_value());
    EXPECT_EQ(*policy.profile_id, "shared-profile");
  }
}

TEST(ConversationConfigTest, LoadMemoryPolicyYamlFile) {
  const std::filesystem::path yaml_path =
      std::filesystem::temp_directory_path() /
      "litert_lm_runtime_memory_policy.yaml";
  {
    std::ofstream out(yaml_path);
    ASSERT_TRUE(out.is_open());
    out << "strategy: raptor\n";
    out << "context_shift_enabled: true\n";
    out << "context_shift_trigger_ratio: 0.75\n";
    out << "context_shift_retain_recent_messages: 6\n";
    out << "context_shift_target_ratio: 0.5\n";
    out << "context_shift_reset_on_exhaustion: false\n";
    out << "context_shift_strategy: drop_all_but_system\n";
  }

  ASSERT_OK_AND_ASSIGN(
      auto policy,
      ConversationConfig::LoadMemoryPolicyYamlFile(yaml_path.string()));
  EXPECT_EQ(policy.strategy,
            ConversationConfig::MemoryStrategy::kIncrementalHierarchicalAggregation);
  EXPECT_TRUE(policy.context_shift_enabled);
  EXPECT_EQ(policy.context_shift_retain_recent_messages, 6);
  EXPECT_EQ(policy.context_shift_strategy,
            ConversationConfig::ContextShiftStrategy::kDropAllButSystem);

  std::error_code ec;
  std::filesystem::remove(yaml_path, ec);
}

TEST(ConversationConfigTest, ParseMemoryPolicyYamlRejectsUnsupportedVersion) {
  const std::string yaml = R"yaml(
strategy: hard_reset_replay_window
version: v2
context_shift_enabled: true
context_shift_trigger_ratio: 0.9
context_shift_retain_recent_messages: 2
context_shift_target_ratio: 0.8
context_shift_reset_on_exhaustion: true
)yaml";
  ASSERT_OK_AND_ASSIGN(auto policy,
                       ConversationConfig::ParseMemoryPolicyYaml(yaml));
  ASSERT_TRUE(policy.version.has_value());
  EXPECT_EQ(*policy.version, "v2");
}

TEST(ConversationConfigTest, ParseMemoryPolicyYamlReadsAllowRuntimeTuning) {
  const std::string yaml = R"yaml(
strategy: hard_reset_replay_window
allow_runtime_tuning: false
context_shift_enabled: true
context_shift_trigger_ratio: 0.9
context_shift_retain_recent_messages: 2
context_shift_target_ratio: 0.8
context_shift_reset_on_exhaustion: true
)yaml";
  ASSERT_OK_AND_ASSIGN(auto policy,
                       ConversationConfig::ParseMemoryPolicyYaml(yaml));
  EXPECT_FALSE(policy.allow_runtime_tuning);
}

struct ConversationTestParams {
  bool enable_constrained_decoding;
  bool prefill_preface_on_init;
};

class ConversationTest : public testing::TestWithParam<ConversationTestParams> {
 public:
  static std::vector<ConversationTestParams> GetTestParams() {
    std::vector<ConversationTestParams> params;
    for (bool enable_constrained_decoding : {true, false}) {
      for (bool prefill_preface_on_init : {true, false}) {
        params.push_back(
            {enable_constrained_decoding, prefill_preface_on_init});
      }
    }
    return params;
  }

 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        tokenizer_,
        SentencePieceTokenizer::CreateFromFile(
            (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
                .string()));
    model_assets_ = ModelAssets::Create(GetTestdataPath(kTestLlmPath));
    ASSERT_OK(model_assets_);
    engine_settings_ =
        EngineSettings::CreateDefault(*model_assets_, Backend::CPU);
    ASSERT_OK(engine_settings_);

    session_config_ = SessionConfig::CreateDefault();
    session_config_.SetStartTokenId(0);
    session_config_.GetMutableStopTokenIds().push_back({1});
    *session_config_.GetMutableLlmModelType().mutable_gemma3() = {};
  }

  std::unique_ptr<MockSession> CreateMockSession() {
    auto mock_session = std::make_unique<MockSession>();
    EXPECT_CALL(*mock_session, GetSessionConfig())
        .WillRepeatedly(testing::ReturnRef(session_config_));
    EXPECT_CALL(*mock_session, GetCurrentStep()).WillRepeatedly(Return(0));
    EXPECT_CALL(*mock_session, GetCacheOpCapabilities())
        .WillRepeatedly(Return(Engine::Session::CacheOpCapabilities{}));
    return mock_session;
  }

  std::unique_ptr<MockEngine> CreateMockEngine(
      std::unique_ptr<MockSession> mock_session) {
    auto mock_engine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mock_engine, GetEngineSettings())
        .WillRepeatedly(testing::ReturnRef(*engine_settings_));
    EXPECT_CALL(*mock_engine, CreateSession(testing::_))
        .WillOnce(testing::Return(std::move(mock_session)));
    EXPECT_CALL(*mock_engine, GetTokenizer())
        .WillRepeatedly(testing::ReturnRef(*tokenizer_));
    return mock_engine;
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  absl::StatusOr<ModelAssets> model_assets_;
  absl::StatusOr<EngineSettings> engine_settings_;
  SessionConfig session_config_ = SessionConfig::CreateDefault();
  bool enable_constrained_decoding_ = GetParam().enable_constrained_decoding;
  bool prefill_preface_on_init_ = GetParam().prefill_preface_on_init;
};

TEST_P(ConversationTest, SendMessage) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  EXPECT_THAT(conversation->GetHistory(), testing::IsEmpty());
  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(user_message));
  ExpectAssistantMessageWithNonEmptyText(message);
  const auto history = conversation->GetHistory();
  ASSERT_THAT(history.size(), 2);
  EXPECT_THAT(history[0], testing::VariantWith<JsonMessage>(user_message));
  EXPECT_THAT(history[1], testing::Eq(message));
}

TEST_P(ConversationTest, SendMessageGemma3Template) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(20);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  std::string gemma3_prompt_template =
      ReadFile(GetTestdataPath(kGemma3TemplatePath));

  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetOverwritePromptTemplate(PromptTemplate(gemma3_prompt_template))
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  EXPECT_THAT(conversation->GetHistory(), testing::IsEmpty());
  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  EXPECT_OK(conversation->SendMessage(user_message));
}

TEST_P(ConversationTest, SendSingleMessage) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_message));

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  EXPECT_EQ(std::get<JsonMessage>(response), assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithExtraContext) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation and overwrite prompt template.
  absl::string_view prompt_template = R"jinja(
{%- if enable_thinking -%}
<start_of_turn>system
Thinking enabled.<end_of_turn>
{% else %}
<start_of_turn>system
Thinking disabled.<end_of_turn>
{%- endif -%}
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(prompt_template))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};
  OptionalArgs optional_args;
  optional_args.extra_context = absl::flat_hash_map<std::string, std::string>{
      {"enable_thinking", "true"}};

  absl::string_view expected_input_text =
      "<start_of_turn>system\nThinking enabled.<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(user_message, std::move(optional_args)));

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  EXPECT_EQ(std::get<JsonMessage>(response), assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithExtraContextOverwritingPreface) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation and overwrite prompt template.
  absl::string_view prompt_template = R"jinja(
{%- if key1 -%}
Key1: {{ key1 + "\n"}}
{%- endif -%}
{%- if key2 -%}
Key2: {{ key2 + "\n"}}
{%- endif -%}
{%- if key3 -%}
Key3: {{ key3 + "\n"}}
{%- endif -%}
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

  JsonPreface preface;

  // This extra context will be set at the Conversation level.
  preface.extra_context = {{"key1", "val1"}, {"key2", "val2"}};

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetPreface(preface)
          .SetOverwritePromptTemplate(PromptTemplate(prompt_template))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message with extra context that overwrites key1 and
  // adds key3.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};
  OptionalArgs optional_args;
  optional_args.extra_context =
      nlohmann::ordered_json{{"key1", "val1_new"}, {"key3", "val3"}};

  // key1 should be overwritten to val1_new.
  // key2 should remain val2.
  // key3 should be added as val3.
  absl::string_view expected_input_text =
      "Key1: val1_new\n"
      "Key2: val2\n"
      "Key3: val3\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(user_message, std::move(optional_args)));

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  EXPECT_EQ(std::get<JsonMessage>(response), assistant_message);
}

TEST_P(ConversationTest, SendMultipleMessages) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send two consecutive messages.
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "Hello world!"
      },
      {
        "role": "user",
        "content": "How are you?"
      }
    ]
  )json");

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "Hello world!<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_messages));

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  EXPECT_EQ(std::get<JsonMessage>(response), assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_messages[0], user_messages[1],
                                   assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithChannel) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetChannels({litert::lm::Channel{
              .channel_name = "thought",
              .start = "<|channel>thought\n",
              .end = "<channel|>",
          }})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(
          Responses(TaskState::kProcessing,
                    {"<|channel>thought\nhmm<channel|>I am good."})));

  // Send the message.
  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_message));

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ],
    "channels": {
      "thought": "hmm"
    }
  })");
  EXPECT_THAT(std::get<JsonMessage>(response), testing::Eq(assistant_message));
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithChannelQwenThink) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetChannels({litert::lm::Channel{
              .channel_name = "thought",
              .start = "<think>\n",
              .end = "\n</think>",
          }})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(Responses(
          TaskState::kProcessing, {"<think>\nhmm\n</think>I am good."})));

  // Send the message.
  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_message));

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ],
    "channels": {
      "thought": "hmm"
    }
  })");
  EXPECT_THAT(std::get<JsonMessage>(response), testing::Eq(assistant_message));
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendMessageWithChannelContentFiltering) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Helper to get the raw text string from `InputText`.
  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetFilterChannelContentFromKvCache(true)
          .SetChannels({litert::lm::Channel{
              .channel_name = "thought",
              .start = "<|channel>thought\n",
              .end = "<channel|>",
          }})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Expect prefill of first user message.
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(Return(absl::OkStatus()));

  // Expect checkpoint to be saved after the first user message is prefilled.
  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("channel_content_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));

  // Expect decode after first user message. Return response with channel
  // content.
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          Return(Responses(TaskState::kProcessing,
                           {"<|channel>thought\nhmm<channel|>I am good."})));

  // Send the first user message.
  JsonMessage user_message_1 = {{"role", "user"}, {"content", "How are you?"}};
  ASSERT_OK(conversation->SendMessage(user_message_1));

  // Expect rewind to checkpoint after second user message is sent.
  EXPECT_CALL(*mock_session_ptr,
              RewindToCheckpoint("channel_content_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));

  // Expect previous assistant message and second user message to be prefilled
  // when the second user message is sent. The assistant message should not
  // have channel content.
  auto assistant_message_matcher =
      AllOf(HasSubstr("I am good."), Not(HasSubstr("hmm")));
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefill(ElementsAre(
          VariantWith<InputText>(ResultOf(get_text, assistant_message_matcher)),
          VariantWith<InputText>(
              ResultOf(get_text, HasSubstr("That's great."))))))
      .WillOnce(Return(absl::OkStatus()));

  // Expect a new checkpoint to be saved after the second user message is
  // prefilled.
  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("channel_content_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));

  // Expect decode after second user message.
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(Return(Responses(TaskState::kProcessing, {"Thank you."})));

  // Send the second user message.
  JsonMessage user_message_2 = {{"role", "user"}, {"content", "That's great."}};
  ASSERT_OK(conversation->SendMessage(user_message_2));
}

TEST_P(ConversationTest, SendMessageWithContextShiftReplay) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  // Keep this small so the trigger ratio produces a tiny threshold.
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(2)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, HasSubstr("How are you?"))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"I am good."})));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(ResultOf(
                    get_text,
                    AllOf(HasSubstr("How are you?"), HasSubstr("I am good.")))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, HasSubstr("That's great."))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"Indeed."})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "How are you?"}}));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "That's great."}}));
}

TEST_P(ConversationTest, SendMessageWithContextShiftDropAllButSystem) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.3f)
          .SetContextShiftRetainRecentMessages(4)
          .SetContextShiftStrategy(
              ConversationConfig::ContextShiftStrategy::kDropAllButSystem)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, HasSubstr("Q1"))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text,
                             AllOf(HasSubstr("Q2"), Not(HasSubstr("Q1")),
                                   Not(HasSubstr("A1"))))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(JsonMessage{{"role", "user"},
                                                  {"content", "Q1"}}));
  ASSERT_OK(conversation->SendMessage(JsonMessage{{"role", "user"},
                                                  {"content", "Q2"}}));
}

TEST_P(ConversationTest, RuntimeMemoryPolicyOverrideEnablesContextShift) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, HasSubstr("Q1"))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));

    EXPECT_CALL(*mock_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text,
                             AllOf(HasSubstr("Q2"), Not(HasSubstr("Q1")),
                                   Not(HasSubstr("A1"))))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  ConversationConfig::RuntimeMemoryPolicy runtime_policy{
      .strategy = ConversationConfig::MemoryStrategy::kHardResetReplayWindow,
      .context_shift_enabled = true,
      .context_shift_trigger_ratio = 0.5f,
      .context_shift_retain_recent_messages = 4,
      .context_shift_target_ratio = 0.3f,
      .context_shift_reset_on_exhaustion = true,
      .context_shift_strategy =
          ConversationConfig::ContextShiftStrategy::kDropAllButSystem,
      .profile_id = std::string("runtime-override"),
  };
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(runtime_policy));

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));
}

TEST_P(ConversationTest,
       SetRuntimeMemoryPolicyQueuesBySafeBoundaryAndAppliesAtMatchingBoundary) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  ConversationConfig::RuntimeMemoryPolicy policy = conversation->GetConfig().runtime_memory_policy();
  policy.safe_boundary = ConversationConfig::SafeBoundary::kToolResult;
  policy.context_shift_enabled = true;
  policy.context_shift_trigger_ratio = 0.5f;
  policy.context_shift_target_ratio = 0.5f;
  policy.context_shift_retain_recent_messages = 4;
  policy.version = std::string("v1");
  policy.compatibility = std::string("v1");
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(policy));
  EXPECT_EQ(conversation->GetQueuedRuntimePolicyUpdateCountForTest(), 1);

  EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  EXPECT_EQ(conversation->GetQueuedRuntimePolicyUpdateCountForTest(), 1);

  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "tool"}, {"content", "tool_result"}}));
  EXPECT_EQ(conversation->GetQueuedRuntimePolicyUpdateCountForTest(), 0);
}

TEST_P(ConversationTest,
       SetRuntimeMemoryPolicyRejectsWhenActivePolicyDisablesRuntimeTuning) {
  auto mock_session = CreateMockSession();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  ConversationConfig::RuntimeMemoryPolicy disable_tuning =
      conversation->GetConfig().runtime_memory_policy();
  disable_tuning.allow_runtime_tuning = false;
  disable_tuning.version = std::string("v1");
  disable_tuning.compatibility = std::string("v1");
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(disable_tuning));

  ConversationConfig::RuntimeMemoryPolicy next_policy =
      conversation->GetConfig().runtime_memory_policy();
  next_policy.version = std::string("v1");
  next_policy.compatibility = std::string("v1");

  auto status = conversation->SetRuntimeMemoryPolicy(next_policy);
  EXPECT_TRUE(absl::IsFailedPrecondition(status));
  EXPECT_THAT(status.message(), HasSubstr("allow_runtime_tuning=false"));
}

TEST_P(ConversationTest,
       SetRuntimeMemoryPolicyRejectsUnsupportedVersionOrCompatibility) {
  auto mock_session = CreateMockSession();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  ConversationConfig::RuntimeMemoryPolicy bad_version =
      conversation->GetConfig().runtime_memory_policy();
  bad_version.version = std::string("v2");
  auto bad_version_status = conversation->SetRuntimeMemoryPolicy(bad_version);
  EXPECT_TRUE(absl::IsInvalidArgument(bad_version_status));
  EXPECT_THAT(bad_version_status.message(),
              HasSubstr("Unsupported runtime policy version"));

  ConversationConfig::RuntimeMemoryPolicy bad_compat =
      conversation->GetConfig().runtime_memory_policy();
  bad_compat.compatibility = std::string("v3");
  auto bad_compat_status = conversation->SetRuntimeMemoryPolicy(bad_compat);
  EXPECT_TRUE(absl::IsFailedPrecondition(bad_compat_status));
  EXPECT_THAT(bad_compat_status.message(),
              HasSubstr("Unsupported runtime policy compatibility"));
}

TEST_P(ConversationTest,
       SetRuntimeMemoryPolicyAppliesAfterAsyncSchedulingFailure) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  EXPECT_CALL(*mock_session_ptr, RunPrefillAsync(testing::_, testing::_))
      .WillOnce(Return(absl::InternalError("schedule failed")));

  absl::Notification done;
  auto status = conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", "Q1"}},
      [&done](absl::StatusOr<Message> message) { done.Notify(); });
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInternal(status));

  ConversationConfig::RuntimeMemoryPolicy policy =
      conversation->GetConfig().runtime_memory_policy();
  policy.context_shift_enabled = true;
  policy.context_shift_trigger_ratio = 0.5f;
  policy.context_shift_target_ratio = 0.5f;
  policy.version = std::string("v1");
  policy.compatibility = std::string("v1");

  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(policy));
}

TEST_P(ConversationTest,
       SetRuntimeMemoryPolicyAppliesAfterAsyncAppendCompletionBoundary) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> append_prefill_callback;
  EXPECT_CALL(*mock_session_ptr, RunPrefillAsync(testing::_, testing::_))
      .WillOnce(
          [&append_prefill_callback](
              const std::vector<InputData>& contents,
              absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            append_prefill_callback = std::move(callback);
            return std::make_unique<MockTaskController>();
          });
  ASSERT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", "Q1"}},
      [](absl::StatusOr<Message> message) {}, {.has_pending_message = true}));

  ConversationConfig::RuntimeMemoryPolicy pending_policy =
      conversation->GetConfig().runtime_memory_policy();
  pending_policy.context_shift_enabled = true;
  pending_policy.context_shift_trigger_ratio = 0.5f;
  pending_policy.context_shift_target_ratio = 0.5f;
  pending_policy.safe_boundary = ConversationConfig::SafeBoundary::kTurnBoundary;
  pending_policy.version = std::string("v1");
  pending_policy.compatibility = std::string("v1");
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(pending_policy));

  ASSERT_TRUE(static_cast<bool>(append_prefill_callback));
  append_prefill_callback(Responses(TaskState::kDone));

  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));
}

TEST_P(ConversationTest,
       SetRuntimeMemoryPolicyEmitsTransitionNoteWhenEnabled) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  ConversationConfig::RuntimeMemoryPolicy policy =
      conversation->GetConfig().runtime_memory_policy();
  policy.version = std::string("v1");
  policy.compatibility = std::string("v1");
  policy.emit_transition_note = true;
  policy.safe_boundary = ConversationConfig::SafeBoundary::kTurnBoundary;

  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(policy));

  const auto history = conversation->GetHistory();
  ASSERT_THAT(history.size(), testing::Eq(1));
  ASSERT_TRUE(std::holds_alternative<JsonMessage>(history[0]));
  const JsonMessage& note = std::get<JsonMessage>(history[0]);
  ASSERT_TRUE(note.is_object());
  EXPECT_EQ(note["role"], "system");
  ASSERT_TRUE(note["content"].is_string());
  EXPECT_THAT(note["content"].get<std::string>(),
              HasSubstr("runtime memory policy transition"));
}

TEST_P(ConversationTest, SendMessageWithContextShiftBudgetShrink) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.3f)
          .SetContextShiftRetainRecentMessages(3)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(
        *mock_session_ptr,
        RunPrefill(ElementsAre(VariantWith<InputText>(
            ResultOf(get_text, AllOf(Not(HasSubstr("Q1")), HasSubstr("A1"),
                                     HasSubstr("Q2"), HasSubstr("A2")))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(6));

    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, AllOf(Not(HasSubstr("Q1")),
                                             HasSubstr("A1"),
                                             HasSubstr("Q2"),
                                             Not(HasSubstr("A2"))))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));

    EXPECT_CALL(*mock_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A3"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q3"}}));
}

TEST_P(ConversationTest, SendMessageWithContextShiftResetOnExhaustion) {
  auto session1 = std::make_unique<MockSession>();
  MockSession* session1_ptr = session1.get();
  EXPECT_CALL(*session1_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config_));

  auto session2 = std::make_unique<MockSession>();
  MockSession* session2_ptr = session2.get();
  EXPECT_CALL(*session2_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config_));

  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(*engine_settings_));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.2f)
          .SetContextShiftRetainRecentMessages(0)
          .SetContextShiftResetOnExhaustion(true)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_engine, CreateSession(testing::_))
        .WillOnce(testing::Return(std::move(session1)));
    EXPECT_CALL(*session1_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*session1_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*session1_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*session1_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));

    EXPECT_CALL(*session1_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*session1_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*session1_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_engine, CreateSession(testing::_))
        .WillOnce(testing::Return(std::move(session2)));
    EXPECT_CALL(*session2_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*session2_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*session2_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));
}

TEST_P(ConversationTest, PolicyUpdateRejectedWhenRuntimeTuningDisabled) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetAllowRuntimeTuning(false)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_)).Times(0);
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_)).Times(0);

  OptionalArgs optional_args;
  optional_args.policy_update_request = ContextShiftPolicyUpdateRequest{
      .profile_schema_version = 1,
      .profile_compatibility_version = 1,
      .runtime_override =
          ContextShiftRuntimePolicyOverride{
              .context_shift_strategy =
                  ConversationConfig::ContextShiftStrategy::kDropAllButSystem},
      .reason = "blocked_update"};

  auto result = conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "hello"}},
      std::move(optional_args));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(absl::IsFailedPrecondition(result.status()));
  EXPECT_THAT(result.status().message(), HasSubstr("allow_runtime_tuning=false"));

  auto records = conversation->GetPolicyTransitionRecordsForTest();
  ASSERT_THAT(records.size(), testing::Eq(1));
  EXPECT_EQ(records[0].action,
            Conversation::PolicyTransitionRecord::Action::kRejected);
  EXPECT_EQ(conversation->GetQueuedPolicyUpdateCountForTest(), 0);
}

TEST_P(ConversationTest, PolicyUpdateRejectsUnsupportedSchemaVersion) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetAllowRuntimeTuning(true)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_)).Times(0);
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_)).Times(0);

  OptionalArgs optional_args;
  optional_args.policy_update_request = ContextShiftPolicyUpdateRequest{
      .profile_schema_version = 7,
      .profile_compatibility_version = 1,
      .runtime_override =
          ContextShiftRuntimePolicyOverride{
              .context_shift_strategy =
                  ConversationConfig::ContextShiftStrategy::kDropAllButSystem},
      .reason = "bad_schema"};

  auto result = conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "hello"}},
      std::move(optional_args));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(result.status()));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Unsupported policy schema version"));

  EXPECT_EQ(conversation->GetActiveContextShiftStrategyForTest(),
            ConversationConfig::ContextShiftStrategy::kReplayRecent);
  auto records = conversation->GetPolicyTransitionRecordsForTest();
  ASSERT_THAT(records.size(), testing::Eq(1));
  EXPECT_EQ(records[0].action,
            Conversation::PolicyTransitionRecord::Action::kRejected);
  EXPECT_EQ(conversation->GetQueuedPolicyUpdateCountForTest(), 0);
}

TEST_P(ConversationTest, PolicyUpdateQueuesThenAppliesOnToolBoundarySync) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetAllowRuntimeTuning(true)
          .SetEmitTransitionNote(true)
          .SetPolicyApplyBoundary(
              ConversationConfig::PolicyApplyBoundary::kToolResult)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
  }

  OptionalArgs first_args;
  first_args.policy_update_request = ContextShiftPolicyUpdateRequest{
      .profile_schema_version = 1,
      .profile_compatibility_version = 1,
      .runtime_override =
          ContextShiftRuntimePolicyOverride{
              .context_shift_strategy =
                  ConversationConfig::ContextShiftStrategy::kDropAllButSystem},
      .reason = "sync_queue_apply"};

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}, std::move(first_args)));
  EXPECT_EQ(conversation->GetQueuedPolicyUpdateCountForTest(), 1);
  EXPECT_EQ(conversation->GetActiveContextShiftStrategyForTest(),
            ConversationConfig::ContextShiftStrategy::kReplayRecent);

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "tool"}, {"content", "tool result"}}));
  EXPECT_EQ(conversation->GetQueuedPolicyUpdateCountForTest(), 0);
  EXPECT_EQ(conversation->GetActiveContextShiftStrategyForTest(),
            ConversationConfig::ContextShiftStrategy::kDropAllButSystem);
  EXPECT_EQ(conversation->GetTransitionNoteCountForTest(), 1);

  auto records = conversation->GetPolicyTransitionRecordsForTest();
  ASSERT_THAT(records.size(), testing::Eq(2));
  EXPECT_EQ(records[0].action,
            Conversation::PolicyTransitionRecord::Action::kQueued);
  EXPECT_EQ(records[1].action,
            Conversation::PolicyTransitionRecord::Action::kApplied);
  EXPECT_EQ(records[0].boundary, "turn_boundary");
  EXPECT_EQ(records[1].boundary, "tool_result");
}

TEST_P(ConversationTest, PolicyUpdateQueuedDuringAsyncTurnAppliesAtTurnBoundary) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetAllowRuntimeTuning(true)
          .SetPolicyApplyBoundary(
              ConversationConfig::PolicyApplyBoundary::kTurnBoundary)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> decode_callback;
  EXPECT_CALL(*mock_session_ptr, RunPrefillAsync(testing::_, testing::_))
      .WillOnce([](const std::vector<InputData>&,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return std::make_unique<MockTaskController>();
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [&decode_callback](
              absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
              const DecodeConfig&) {
            decode_callback = std::move(user_callback);
            return std::make_unique<MockTaskController>();
          });

  OptionalArgs optional_args;
  optional_args.policy_update_request = ContextShiftPolicyUpdateRequest{
      .profile_schema_version = 1,
      .profile_compatibility_version = 1,
      .runtime_override =
          ContextShiftRuntimePolicyOverride{
              .context_shift_strategy =
                  ConversationConfig::ContextShiftStrategy::kDropAllButSystem},
      .reason = "async_turn_boundary_apply"};

  Message expected_message = JsonMessage(
      {{"role", "assistant"},
       {"content", {{{"type", "text"}, {"text", "async result"}}}}});
  Message expected_message_for_confirm = expected_message;
  absl::Notification done;
  ASSERT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "tool"}, {"content", "tool result"}},
      CreateTestMessageCallback(expected_message, done), std::move(optional_args)));

  EXPECT_EQ(conversation->GetQueuedPolicyUpdateCountForTest(), 1);
  EXPECT_EQ(conversation->GetActiveContextShiftStrategyForTest(),
            ConversationConfig::ContextShiftStrategy::kReplayRecent);
  auto before_records = conversation->GetPolicyTransitionRecordsForTest();
  ASSERT_THAT(before_records.size(), testing::Eq(1));
  EXPECT_EQ(before_records[0].action,
            Conversation::PolicyTransitionRecord::Action::kQueued);

  ASSERT_TRUE(static_cast<bool>(decode_callback));
  decode_callback(Responses(TaskState::kProcessing, {"async result"}));
  decode_callback(Responses(TaskState::kDone));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_EQ(conversation->GetQueuedPolicyUpdateCountForTest(), 0);
  EXPECT_EQ(conversation->GetActiveContextShiftStrategyForTest(),
            ConversationConfig::ContextShiftStrategy::kDropAllButSystem);
  auto records = conversation->GetPolicyTransitionRecordsForTest();
  ASSERT_THAT(records.size(), testing::Eq(2));
  EXPECT_EQ(records[1].action,
            Conversation::PolicyTransitionRecord::Action::kApplied);
  EXPECT_EQ(records[1].boundary, "turn_boundary");
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(
                  JsonMessage{{"role", "tool"}, {"content", "tool result"}},
                  expected_message_for_confirm));
}

TEST_P(ConversationTest, PrefetchPlannerMetricsIncrementInShadowMode) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(true)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.planned_count, 1);
  EXPECT_EQ(metrics.install_attempt_count, 0);
  EXPECT_EQ(metrics.install_hit_count, 0);
  EXPECT_EQ(metrics.shadow_skip_count, 0);
  EXPECT_EQ(metrics.install_failure_count, 0);
  EXPECT_EQ(metrics.fallback_count, 0);
  EXPECT_EQ(metrics.parity_check_count, 0);
  EXPECT_EQ(metrics.parity_mismatch_count, 0);

  const auto planner = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(planner.lifecycle_state,
            Conversation::PrefetchLifecycleState::kReady);
  EXPECT_EQ(planner.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kNone);
  EXPECT_GT(planner.active_plan_token, 0u);
  EXPECT_GT(planner.last_confidence_score, 0.0f);
}

TEST_P(ConversationTest, PrefetchFallbackMetricsIncrementOnContextShift) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(0)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(true)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.planned_count, 1);
  EXPECT_EQ(metrics.install_attempt_count, 1);
  EXPECT_EQ(metrics.install_hit_count, 0);
  EXPECT_EQ(metrics.stale_discard_count, 0);
  EXPECT_EQ(metrics.shadow_skip_count, 1);
  EXPECT_EQ(metrics.install_failure_count, 0);
  EXPECT_EQ(metrics.fallback_count, 1);
  EXPECT_EQ(metrics.parity_check_count, 1);
  EXPECT_EQ(metrics.parity_mismatch_count, 0);
  EXPECT_GE(metrics.baseline_recompute_latency_ms_total, 0.0);

  const auto planner = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(planner.lifecycle_state,
            Conversation::PrefetchLifecycleState::kDiscarded);
  EXPECT_EQ(planner.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kShadowMode);
}

TEST(ConversationCacheOpsVocabularyTest,
     NativeCacheVocabularyMatchesPhaseCRfcDraft) {
  using Session = Engine::Session;

  EXPECT_EQ(Conversation::NativeCacheOpVerbToString(Session::CacheOpVerb::kPin),
            "Pin");
  EXPECT_EQ(
      Conversation::NativeCacheOpVerbToString(Session::CacheOpVerb::kEvictRange),
      "EvictRange");
  EXPECT_EQ(
      Conversation::NativeCacheFailureCodeToString(
          Session::CacheOpFailureCode::kRollbackUnavailable),
      "rollback_unavailable");
  EXPECT_EQ(
      Conversation::NativeCacheFailureCodeToString(
          Session::CacheOpFailureCode::kInternalCacheCorruptionSuspected),
      "internal_cache_corruption_suspected");

  Session::CacheBlockMetadata metadata{
      .block_id = {.session_epoch = 9, .block_seqno = 42},
      .token_span = {.start_token = 12, .end_token_exclusive = 24},
      .pin_class = Session::CachePinClass::kAttentionSink,
      .logical_role = Session::CacheLogicalRole::kSummaryAnchor,
  };
  EXPECT_EQ(metadata.block_id.session_epoch, 9);
  EXPECT_EQ(metadata.block_id.block_seqno, 42);
  EXPECT_EQ(metadata.token_span.start_token, 12);
  EXPECT_EQ(metadata.token_span.end_token_exclusive, 24);
  EXPECT_EQ(Conversation::NativeCachePinClassToString(metadata.pin_class),
            "attention_sink");
  EXPECT_EQ(Conversation::NativeCacheLogicalRoleToString(metadata.logical_role),
            "summary_anchor");
}

TEST_P(ConversationTest,
       NativeCacheCapabilitiesImmutableAndNativeCommitSkipsPhaseBFallback) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  const Engine::Session::CacheOpCapabilities capabilities{
      .supports_kv_surgery = true,
      .supports_attention_sink_pinning = true,
      .supports_range_evict = true,
      .supports_block_remap = false,
  };
  EXPECT_CALL(*mock_session_ptr, GetCacheOpCapabilities())
      .Times(1)
      .WillOnce(Return(capabilities));
  EXPECT_CALL(*mock_session_ptr,
              RewindToCheckpoint("context_shift_anchor_checkpoint"))
      .Times(0);

  auto mock_engine = CreateMockEngine(std::move(mock_session));
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(0)
          .SetPrefetchEnabled(false)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr, ExecuteCacheOpGroup(testing::_))
        .WillOnce([](const Engine::Session::CacheOpGroup& op_group) {
          EXPECT_THAT(op_group.ops, SizeIs(1));
          EXPECT_EQ(op_group.ops[0].verb, Engine::Session::CacheOpVerb::kEvictRange);
          EXPECT_EQ(op_group.ops[0].token_span.start_token, 0);
          EXPECT_EQ(op_group.ops[0].token_span.end_token_exclusive, 3);
          EXPECT_TRUE(op_group.requires_rollback_guarantee);
          return Engine::Session::CacheOpGroupResult{
              .committed = true, .rollback_available = true};
        });
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  const auto discovered = conversation->GetNativeCacheCapabilitiesForTest();
  EXPECT_TRUE(discovered.supports_kv_surgery);
  EXPECT_TRUE(discovered.supports_range_evict);
  EXPECT_TRUE(discovered.supports_attention_sink_pinning);

  const auto native_state = conversation->GetNativeCacheStateForTest();
  EXPECT_TRUE(native_state.attempted);
  EXPECT_TRUE(native_state.committed);
  EXPECT_FALSE(native_state.fallback_to_phase_b);
  EXPECT_FALSE(native_state.last_failure_code.has_value());
}

TEST_P(ConversationTest,
       NativeCacheCapabilityGatingKeepsUnsupportedSessionsOnPhaseBPath) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  const Engine::Session::CacheOpCapabilities capabilities{
      .supports_kv_surgery = false,
      .supports_range_evict = true,
  };
  EXPECT_CALL(*mock_session_ptr, GetCacheOpCapabilities())
      .Times(1)
      .WillOnce(Return(capabilities));
  EXPECT_CALL(*mock_session_ptr, ExecuteCacheOpGroup(testing::_)).Times(0);

  auto mock_engine = CreateMockEngine(std::move(mock_session));
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(0)
          .SetPrefetchEnabled(false)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  const auto discovered = conversation->GetNativeCacheCapabilitiesForTest();
  EXPECT_FALSE(discovered.supports_kv_surgery);
  EXPECT_TRUE(discovered.supports_range_evict);

  const auto native_state = conversation->GetNativeCacheStateForTest();
  EXPECT_FALSE(native_state.attempted);
  EXPECT_FALSE(native_state.committed);
  EXPECT_FALSE(native_state.fallback_to_phase_b);
  EXPECT_FALSE(native_state.last_failure_code.has_value());
}

TEST_P(ConversationTest,
       NativeCacheUnsupportedCapabilityTriggersDeterministicFallback) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  const Engine::Session::CacheOpCapabilities capabilities{
      .supports_kv_surgery = true,
      .supports_range_evict = true,
  };
  EXPECT_CALL(*mock_session_ptr, GetCacheOpCapabilities())
      .Times(1)
      .WillOnce(Return(capabilities));

  auto mock_engine = CreateMockEngine(std::move(mock_session));
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(0)
          .SetPrefetchEnabled(false)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr, ExecuteCacheOpGroup(testing::_))
        .WillOnce(Return(
            absl::UnimplementedError("native cache ops unavailable in engine")));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  const auto native_state = conversation->GetNativeCacheStateForTest();
  ASSERT_TRUE(native_state.last_failure_code.has_value());
  EXPECT_TRUE(native_state.attempted);
  EXPECT_FALSE(native_state.committed);
  EXPECT_TRUE(native_state.fallback_to_phase_b);
  EXPECT_EQ(*native_state.last_failure_code,
            Engine::Session::CacheOpFailureCode::kUnsupportedCapability);
}

TEST_P(ConversationTest, NativeCacheRollbackUnavailableForcesFallback) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  const Engine::Session::CacheOpCapabilities capabilities{
      .supports_kv_surgery = true,
      .supports_range_evict = true,
  };
  EXPECT_CALL(*mock_session_ptr, GetCacheOpCapabilities())
      .Times(1)
      .WillOnce(Return(capabilities));

  auto mock_engine = CreateMockEngine(std::move(mock_session));
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(0)
          .SetPrefetchEnabled(false)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr, ExecuteCacheOpGroup(testing::_))
        .WillOnce(Return(Engine::Session::CacheOpGroupResult{
            .committed = false,
            .rollback_available = false,
            .failure = Engine::Session::CacheOpFailure{
                .code = Engine::Session::CacheOpFailureCode::kRangeConflict,
                .detail = "partial mutation detected"},
        }));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  const auto native_state = conversation->GetNativeCacheStateForTest();
  ASSERT_TRUE(native_state.last_failure_code.has_value());
  EXPECT_TRUE(native_state.attempted);
  EXPECT_FALSE(native_state.committed);
  EXPECT_TRUE(native_state.fallback_to_phase_b);
  EXPECT_EQ(*native_state.last_failure_code,
            Engine::Session::CacheOpFailureCode::kRollbackUnavailable);
}

TEST_P(ConversationTest, NativeCacheCorruptionSignalForcesFallback) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  const Engine::Session::CacheOpCapabilities capabilities{
      .supports_kv_surgery = true,
      .supports_range_evict = true,
  };
  EXPECT_CALL(*mock_session_ptr, GetCacheOpCapabilities())
      .Times(1)
      .WillOnce(Return(capabilities));

  auto mock_engine = CreateMockEngine(std::move(mock_session));
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(0)
          .SetPrefetchEnabled(false)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr, ExecuteCacheOpGroup(testing::_))
        .WillOnce(Return(Engine::Session::CacheOpGroupResult{
            .committed = false,
            .rollback_available = true,
            .failure = Engine::Session::CacheOpFailure{
                .code = Engine::Session::CacheOpFailureCode::
                    kInternalCacheCorruptionSuspected,
                .detail = "allocator mismatch"},
        }));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  const auto native_state = conversation->GetNativeCacheStateForTest();
  ASSERT_TRUE(native_state.last_failure_code.has_value());
  EXPECT_TRUE(native_state.attempted);
  EXPECT_FALSE(native_state.committed);
  EXPECT_TRUE(native_state.fallback_to_phase_b);
  EXPECT_EQ(*native_state.last_failure_code,
            Engine::Session::CacheOpFailureCode::
                kInternalCacheCorruptionSuspected);
}

TEST_P(ConversationTest, PrefetchReplayPackInstallsOnBoundaryWhenValid) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(2)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(false)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, AllOf(HasSubstr("Q1"), HasSubstr("A1")))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.planned_count, 1);
  EXPECT_EQ(metrics.install_attempt_count, 1);
  EXPECT_EQ(metrics.install_hit_count, 1);
  EXPECT_EQ(metrics.shadow_skip_count, 0);
  EXPECT_EQ(metrics.install_failure_count, 0);
  EXPECT_EQ(metrics.fallback_count, 0);
  EXPECT_GE(metrics.install_latency_ms_total, 0.0);

  const auto planner = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(planner.lifecycle_state,
            Conversation::PrefetchLifecycleState::kInstalled);
  EXPECT_EQ(planner.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kNone);
  EXPECT_EQ(planner.last_successful_install_step, 5);
}

TEST_P(ConversationTest, PrefetchFallbackMetricsNotCountedWithoutInstallAttempt) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(0)
          .SetPrefetchEnabled(false)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.install_attempt_count, 0);
  EXPECT_EQ(metrics.install_hit_count, 0);
  EXPECT_EQ(metrics.shadow_skip_count, 0);
  EXPECT_EQ(metrics.install_failure_count, 0);
  EXPECT_EQ(metrics.fallback_count, 0);
}

TEST_P(ConversationTest, PrefetchPlannerReusesExistingUsefulPlan) {
  auto mock_session = CreateMockSession();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(1)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(true)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));
  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));
  }
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  const auto planned = conversation->GetPrefetchPlannerStateForTest();
  ASSERT_GT(planned.active_plan_token, 0u);

  conversation->MaybePlanPrefetchPackForTest(3);
  const auto reused = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(reused.active_plan_token, planned.active_plan_token);
  EXPECT_EQ(reused.lifecycle_state,
            Conversation::PrefetchLifecycleState::kReady);
  EXPECT_EQ(reused.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kExistingPlanStillUseful);

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.planned_count, 1);
}

TEST_P(ConversationTest, PrefetchInstallDiscardedOnRetainedSliceDigestMismatch) {
  auto mock_session = CreateMockSession();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(1)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(false)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));
  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(4));
  }
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  ASSERT_OK(conversation->ReplaceHistoryMessageForTest(
      1, JsonMessage{{"role", "assistant"}, {"content", "A1-mutated"}}, false));

  ASSERT_OK_AND_ASSIGN(auto outcome,
                       conversation->TryInstallPrefetchPackForTest(5));
  EXPECT_EQ(outcome, Conversation::PrefetchInstallOutcome::kStaleDiscarded);

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.install_attempt_count, 1);
  EXPECT_EQ(metrics.stale_discard_count, 1);
  EXPECT_EQ(metrics.install_hit_count, 0);

  const auto planner = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(planner.lifecycle_state,
            Conversation::PrefetchLifecycleState::kDiscarded);
  EXPECT_EQ(planner.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kRetainedSliceChanged);
}

TEST_P(ConversationTest, PrefetchPlanDiscardedWhenRuntimePolicyChanges) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(1)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(false)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));
  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(4));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
  }
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  ConversationConfig::RuntimeMemoryPolicy updated_policy =
      conversation->GetConfig().runtime_memory_policy();
  updated_policy.context_shift_enabled = true;
  updated_policy.context_shift_trigger_ratio = 0.8f;
  updated_policy.context_shift_target_ratio = 0.4f;
  updated_policy.context_shift_retain_recent_messages = 2;
  updated_policy.safe_boundary = ConversationConfig::SafeBoundary::kTurnBoundary;
  updated_policy.version = std::string("v1");
  updated_policy.compatibility = std::string("v1");
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(updated_policy));

  ASSERT_OK_AND_ASSIGN(auto outcome,
                       conversation->TryInstallPrefetchPackForTest(4));
  EXPECT_EQ(outcome, Conversation::PrefetchInstallOutcome::kNoPendingPack);

  const auto planner = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(planner.lifecycle_state,
            Conversation::PrefetchLifecycleState::kDiscarded);
  EXPECT_EQ(planner.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kPolicyChanged);
}

TEST_P(ConversationTest, PrefetchPlannerRunsAsynchronouslyAfterBoundary) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(1)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(true)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  EXPECT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.planned_count, 1);

  const auto planner = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(planner.lifecycle_state,
            Conversation::PrefetchLifecycleState::kReady);
  EXPECT_GT(planner.active_plan_token, 0u);
}

TEST_P(ConversationTest, PrefetchReadyPackCarriesBuilderIdentityMetadata) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(2)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(true)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  const auto planner = conversation->GetPrefetchPlannerStateForTest();
  EXPECT_EQ(planner.lifecycle_state,
            Conversation::PrefetchLifecycleState::kReady);
  EXPECT_EQ(planner.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kNone);
}

TEST_P(ConversationTest, PrefetchMetricsCaptureStructuredDimensions) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(1)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(true)
          .SetPrefetchRatio(0.2f)
          .SetMemoryStrategy(
              ConversationConfig::MemoryStrategy::kSummarizeProtectedTail)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  auto policy = conversation->GetConfig().runtime_memory_policy();
  policy.profile_id = std::string("phase-b-profile");
  policy.version = std::string("v1");
  policy.compatibility = std::string("v1");
  ASSERT_OK(conversation->SetRuntimeMemoryPolicy(policy));

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  ASSERT_FALSE(metrics.events.empty());
  const auto& last_event = metrics.events.back();
  EXPECT_EQ(last_event.outcome,
            Conversation::PrefetchMetrics::Outcome::kPlanned);
  EXPECT_EQ(last_event.dimensions.profile_id, "phase-b-profile");
  EXPECT_EQ(last_event.dimensions.strategy, "summarize_protected_tail");
  EXPECT_EQ(last_event.dimensions.builder_id, "summarize_protected_tail");
  EXPECT_EQ(last_event.dimensions.model_type, "gemma3");
  EXPECT_EQ(last_event.dimensions.reason_code, "planned");
  EXPECT_EQ(last_event.parity_mode,
            Conversation::PrefetchParityMode::kSemanticParity);
  EXPECT_TRUE(last_event.scaffold_only);
}

TEST_P(ConversationTest, SupersedingQueuedPlanRemovesOlderPendingTask) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.9f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(1)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(true)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));

  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  const auto first_plan = conversation->GetPrefetchPlannerStateForTest();

  ASSERT_OK(conversation->ReplaceHistoryMessageForTest(
      1, JsonMessage{{"role", "assistant"}, {"content", "A1-mutated"}}, true));
  conversation->MaybePlanPrefetchPackForTest(4);

  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  const auto second_plan = conversation->GetPrefetchPlannerStateForTest();

  EXPECT_GT(second_plan.active_plan_token, first_plan.active_plan_token);
  EXPECT_EQ(second_plan.last_invalidation_reason,
            Conversation::PrefetchInvalidationReason::kNone);

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.planned_count, 2);
}

TEST_P(ConversationTest, PrefetchLongSessionInstallHitsAcrossMultipleTurns) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(2)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(false)
          .SetPrefetchRatio(0.2f)
          .Build(*mock_engine));

  {
    InSequence seq;
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(3));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, AllOf(HasSubstr("Q1"), HasSubstr("A1")))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(4));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(ResultOf(
                    get_text, AllOf(HasSubstr("Q2"), HasSubstr("A2"),
                                    Not(HasSubstr("Q1")),
                                    Not(HasSubstr("A1"))))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A3"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(4));

    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*mock_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(ResultOf(
                    get_text, AllOf(HasSubstr("Q3"), HasSubstr("A3"),
                                    Not(HasSubstr("Q2")),
                                    Not(HasSubstr("A2"))))))))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A4"})));
    EXPECT_CALL(*mock_session_ptr, GetCurrentStep()).WillOnce(Return(4));
  }

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q3"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q4"}}));
  ASSERT_TRUE(conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));

  const auto metrics = conversation->GetPrefetchMetricsForTest();
  EXPECT_EQ(metrics.planned_count, 4);
  EXPECT_EQ(metrics.install_attempt_count, 3);
  EXPECT_EQ(metrics.install_hit_count, 3);
  EXPECT_EQ(metrics.fallback_count, 0);
  EXPECT_GT(metrics.install_latency_ms_total, 0.0);
}

TEST_P(ConversationTest,
       PrefetchInstallHitPathRecordsLowerLatencyThanBaselineRecompute) {
  auto install_session = CreateMockSession();
  MockSession* install_session_ptr = install_session.get();
  auto install_engine = CreateMockEngine(std::move(install_session));

  auto baseline_session = CreateMockSession();
  MockSession* baseline_session_ptr = baseline_session.get();
  auto baseline_engine = CreateMockEngine(std::move(baseline_session));

  engine_settings_->GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  ASSERT_OK_AND_ASSIGN(
      auto install_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(2)
          .SetPrefetchEnabled(true)
          .SetPrefetchShadowMode(false)
          .SetPrefetchRatio(0.2f)
          .Build(*install_engine));

  ASSERT_OK_AND_ASSIGN(
      auto baseline_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableContextShift(true)
          .SetContextShiftTriggerRatio(0.5f)
          .SetContextShiftTargetRatio(0.5f)
          .SetContextShiftRetainRecentMessages(2)
          .SetPrefetchEnabled(false)
          .Build(*baseline_engine));

  {
    InSequence seq;
    EXPECT_CALL(*install_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*install_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*install_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*install_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));
    EXPECT_CALL(*install_session_ptr, GetCurrentStep()).WillOnce(Return(3));

    EXPECT_CALL(*install_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*install_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*install_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, AllOf(HasSubstr("Q1"), HasSubstr("A1")))))))
        .WillOnce([](const std::vector<InputData>&) {
          absl::SleepFor(absl::Milliseconds(5));
          return absl::OkStatus();
        });
    EXPECT_CALL(*install_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*install_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*install_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
    EXPECT_CALL(*install_session_ptr, GetCurrentStep()).WillOnce(Return(0));
  }

  {
    InSequence seq;
    EXPECT_CALL(*baseline_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*baseline_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*baseline_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*baseline_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A1"})));

    EXPECT_CALL(*baseline_session_ptr, GetCurrentStep()).WillOnce(Return(8));
    EXPECT_CALL(*baseline_session_ptr,
                RewindToCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*baseline_session_ptr,
                RunPrefill(ElementsAre(VariantWith<InputText>(
                    ResultOf(get_text, AllOf(HasSubstr("Q1"), HasSubstr("A1")))))))
        .WillOnce([](const std::vector<InputData>&) {
          absl::SleepFor(absl::Milliseconds(40));
          return absl::OkStatus();
        });
    EXPECT_CALL(*baseline_session_ptr, GetCurrentStep()).WillOnce(Return(0));
    EXPECT_CALL(*baseline_session_ptr,
                SaveCheckpoint("context_shift_anchor_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*baseline_session_ptr, RunPrefill(testing::_))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*baseline_session_ptr, RunDecode(testing::_))
        .WillOnce(Return(Responses(TaskState::kProcessing, {"A2"})));
  }

  ASSERT_OK_AND_ASSIGN(auto install_conversation,
                       Conversation::Create(*install_engine, install_config));
  ASSERT_OK_AND_ASSIGN(auto baseline_conversation,
                       Conversation::Create(*baseline_engine, baseline_config));

  ASSERT_OK(install_conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_TRUE(install_conversation->WaitForPrefetchPlannerStateForTest(
      Conversation::PrefetchLifecycleState::kReady));
  ASSERT_OK(install_conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));

  ASSERT_OK(baseline_conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q1"}}));
  ASSERT_OK(baseline_conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Q2"}}));

  const auto install_metrics = install_conversation->GetPrefetchMetricsForTest();
  const auto baseline_metrics =
      baseline_conversation->GetPrefetchMetricsForTest();

  EXPECT_GT(install_metrics.install_latency_ms_total, 0.0);
  EXPECT_GT(baseline_metrics.baseline_recompute_latency_ms_total, 0.0);
  EXPECT_GT(baseline_metrics.baseline_recompute_latency_ms_total,
            install_metrics.install_latency_ms_total + 10.0);
}

TEST_P(ConversationTest, SendMultipleMessagesWithHistory) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // The first user message.
  JsonMessage user_message_1 = nlohmann::ordered_json::parse(R"json(
    {
      "role": "user",
      "content": "How are you?"
    }
  )json");
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));

  // The first assistant response.
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  // Send the first user message to fill the history.
  ASSERT_OK(conversation->SendMessage(user_message_1));
  ASSERT_THAT(conversation->GetHistory().size(), testing::Eq(2));

  // We will send two consecutive messages when the history is not empty.
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "foo"
      },
      {
        "role": "user",
        "content": "bar"
      }
    ]
  )json");
  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "foo<end_of_turn>\n"
      "<start_of_turn>user\n"
      "bar<end_of_turn>\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));

  // The second assistant response.
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(Responses(TaskState::kProcessing, {"baz"})));

  // Send the user messages.
  ASSERT_OK(conversation->SendMessage(user_messages));

  // Check the history.
  JsonMessage assistant_message_1 = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  JsonMessage assistant_message_2 = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "baz"
      }
    ]
  })");
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message_1, assistant_message_1,
                                   user_messages[0], user_messages[1],
                                   assistant_message_2));
}

TEST_P(ConversationTest, RunTextScoring) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Test sync scoring.
  auto cloned_session_sync = std::make_unique<MockSession>();
  EXPECT_CALL(*cloned_session_sync,
              RunTextScoring(testing::ElementsAre("I am good."), true))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));
  EXPECT_CALL(*mock_session_ptr, Clone())
      .WillOnce(testing::Return(std::move(cloned_session_sync)));

  ASSERT_OK_AND_ASSIGN(const Responses response,
                       conversation->RunTextScoring({"I am good."}));
  EXPECT_EQ(response.GetTexts()[0], "I am good.");

  // Test async scoring.
  auto cloned_session_async = std::make_unique<MockSession>();
  EXPECT_CALL(
      *cloned_session_async,
      RunTextScoringAsync(testing::ElementsAre("I am good."), testing::_, true))
      .WillOnce([](const std::vector<absl::string_view>& target_text,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
                   bool store_token_lengths) {
        callback(Responses(TaskState::kProcessing, {"I am good."}));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, CloneAsync(testing::_))
      .WillOnce(testing::Return(std::move(cloned_session_async)));

  absl::Notification done;
  std::string response_text;
  EXPECT_OK(conversation->RunTextScoringAsync(
      {"I am good."}, [&](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        response_text = responses->GetTexts()[0];
        done.Notify();
      }));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));
  EXPECT_EQ(response_text, "I am good.");
}

TEST_P(ConversationTest, SendMessageAsync) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  int partial_message_count = 0;
  absl::Notification done;
  EXPECT_OK(conversation->SendMessageAsync(
      user_message, CreateStreamingObserverCallback(partial_message_count, done)));
  // Wait for the async message to be processed.
  EXPECT_OK(engine->WaitUntilDone(absl::Seconds(100)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));
  EXPECT_GT(partial_message_count, 0);
  const auto history = conversation->GetHistory();
  ASSERT_THAT(history.size(), 2);
  EXPECT_THAT(history[0], testing::VariantWith<JsonMessage>(user_message));
  ExpectAssistantMessageWithNonEmptyText(history[1]);
}

TEST_P(ConversationTest, SendSingleMessageAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message = JsonMessage(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })"));
  Message assistant_message_for_confirm = assistant_message;
  absl::Notification done;
  auto message_callback = CreateTestMessageCallback(assistant_message, done);
  EXPECT_OK(conversation->SendMessageAsync(user_message,
                                           std::move(message_callback)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(
      conversation->GetHistory(),
      testing::ElementsAre(user_message, assistant_message_for_confirm));
}

TEST_P(ConversationTest, SendMessageAsyncWithChannelContent) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  std::vector<Channel> custom_channels = {{"thought", "<think>", "</think>"}};
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetChannels(custom_channels)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });

  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing,
                                    {"Hello <think>hmm</think> World!"}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  absl::Notification done;

  std::vector<Message> expected_messages = {
      JsonMessage{{"role", "assistant"},
                  {"content", {{{"type", "text"}, {"text", "Hello "}}}}},
      JsonMessage{{"role", "assistant"}, {"channels", {{"thought", "hmm"}}}},
      JsonMessage{{"role", "assistant"},
                  {"content", {{{"type", "text"}, {"text", " World!"}}}}},
  };
  auto message_callback =
      CreateTestMultiMessageCallback(expected_messages, done);
  EXPECT_OK(conversation->SendMessageAsync(user_message,
                                           std::move(message_callback)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  // Verify the final message in history.
  JsonMessage expected_assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "Hello  World!"
      }
    ],
    "channels": {
      "thought": "hmm"
    }
  })");

  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, expected_assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageAsyncWithExtraContext) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation and overwrite prompt template.
  absl::string_view prompt_template = R"jinja(
{%- if enable_thinking -%}
<start_of_turn>system
Thinking enabled.<end_of_turn>
{% else %}
<start_of_turn>system
Thinking disabled.<end_of_turn>
{%- endif -%}
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(prompt_template))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};
  OptionalArgs optional_args;
  optional_args.extra_context = absl::flat_hash_map<std::string, std::string>{
      {"enable_thinking", "true"}};

  absl::string_view expected_input_text =
      "<start_of_turn>system\nThinking enabled.<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(
                Responses(TaskState::kProcessing, {"I am good async."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message = JsonMessage(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good async."
      }
    ]
  })"));
  Message assistant_message_for_confirm = assistant_message;
  absl::Notification done;
  auto message_callback = CreateTestMessageCallback(assistant_message, done);
  EXPECT_OK(conversation->SendMessageAsync(
      user_message, std::move(message_callback), std::move(optional_args)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(
      conversation->GetHistory(),
      testing::ElementsAre(user_message, assistant_message_for_confirm));
}

TEST_P(ConversationTest, SendMultipleMessagesAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send two consecutive messages.
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "Hello world!"
      },
      {
        "role": "user",
        "content": "How are you?"
      }
    ]
  )json");

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "Hello world!<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message = JsonMessage(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })json"));
  Message assistant_message_for_confirm = assistant_message;
  absl::Notification done;
  auto message_callback = CreateTestMessageCallback(assistant_message, done);
  EXPECT_OK(conversation->SendMessageAsync(user_messages,
                                           std::move(message_callback)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_messages[0], user_messages[1],
                                   assistant_message_for_confirm));
}

TEST_P(ConversationTest, SendMessageAsyncWithChannelContentFiltering) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation with channel content filtering enabled.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetFilterChannelContentFromKvCache(true)
          .SetChannels({litert::lm::Channel{
              .channel_name = "thought",
              .start = "<|channel>thought\n",
              .end = "<channel|>",
          }})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Helper to get the raw text string from `InputText`.
  auto get_text = [](const InputText& it) -> std::string {
    auto status_or_view = it.GetRawTextString();
    if (!status_or_view.ok()) return "";
    return std::string(*status_or_view);
  };

  // Expect prefill of first user message.
  EXPECT_CALL(*mock_session_ptr, RunPrefillAsync(testing::_, testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });

  // Expect checkpoint to be saved.
  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("channel_content_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));

  // Expect decode after first user message.
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(
                Responses(TaskState::kProcessing,
                          {"<|channel>thought\nhmm<channel|>I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  // Prepare the first user message.
  JsonMessage user_message_1 = {{"role", "user"}, {"content", "How are you?"}};
  Message assistant_message_1 =
      JsonMessage(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })json"));
  absl::Notification done_1;
  auto message_callback_1 = [&done_1](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      done_1.Notify();
      return;
    }
    if (auto* json_msg = std::get_if<JsonMessage>(&message.value())) {
      if (json_msg->is_null()) {
        done_1.Notify();
      }
    }
  };

  // Send the first user message.
  EXPECT_OK(conversation->SendMessageAsync(user_message_1,
                                           std::move(message_callback_1)));
  ASSERT_TRUE(done_1.WaitForNotificationWithTimeout(absl::Seconds(10)));

  // Prepare the second user message.
  JsonMessage user_message_2 = {{"role", "user"}, {"content", "That's great."}};
  absl::Notification done_2;
  Message assistant_message_2 =
      JsonMessage(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "Indeed."
      }
    ]
  })json"));
  auto message_callback_2 = [&done_2](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      done_2.Notify();
      return;
    }
    if (auto* json_msg = std::get_if<JsonMessage>(&message.value())) {
      if (json_msg->is_null()) {
        done_2.Notify();
      }
    }
  };

  // Expect rewind to checkpoint when second user message is sent.
  EXPECT_CALL(*mock_session_ptr,
              RewindToCheckpoint("channel_content_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));

  // Expect the previous assistant message and the new user message to be
  // prefilled asynchronously. The previous assistant message should not contain
  // channel content.
  auto assistant_message_matcher =
      AllOf(HasSubstr("I am good."), Not(HasSubstr("hmm")));
  auto message_input_matcher = ElementsAre(
      VariantWith<InputText>(ResultOf(get_text, assistant_message_matcher)),
      VariantWith<InputText>(ResultOf(get_text, HasSubstr("That's great."))));
  EXPECT_CALL(*mock_session_ptr,
              RunPrefillAsync(message_input_matcher, testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });

  // Expect a new checkpoint to be saved before decode of second turn.
  EXPECT_CALL(*mock_session_ptr, SaveCheckpoint("channel_content_checkpoint"))
      .WillOnce(Return(absl::OkStatus()));

  // Expect decode after second user message.
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"Indeed."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  // Send the second user message.
  EXPECT_OK(conversation->SendMessageAsync(user_message_2,
                                           std::move(message_callback_2)));
  ASSERT_TRUE(done_2.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

TEST_P(ConversationTest, SendMultipleMessagesAsyncWithHistory) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // The first user message.
  JsonMessage user_message_1 = nlohmann::ordered_json::parse(R"json(
    {
      "role": "user",
      "content": "How are you?"
    }
  )json");
  absl::string_view expected_input_text1 =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text1))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message_1 =
      JsonMessage(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })json"));
  Message assistant_message_1_for_confirm = assistant_message_1;

  absl::Notification done_1;
  EXPECT_OK(conversation->SendMessageAsync(
      user_message_1, CreateTestMessageCallback(assistant_message_1, done_1)));
  done_1.WaitForNotificationWithTimeout(absl::Seconds(10));
  ASSERT_THAT(conversation->GetHistory().size(), testing::Eq(2));

  // We will send two consecutive messages when the history is not empty.
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "foo"
      },
      {
        "role": "user",
        "content": "bar"
      }
    ]
  )json");

  absl::string_view expected_input_text2 =
      "<start_of_turn>user\n"
      "foo<end_of_turn>\n"
      "<start_of_turn>user\n"
      "bar<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text2))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"baz"}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message_2 =
      JsonMessage(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "baz"
      }
    ]
  })json"));
  Message assistant_message_2_for_confirm = assistant_message_2;

  absl::Notification done_2;
  auto message_callbacks_2 =
      CreateTestMessageCallback(assistant_message_2, done_2);
  EXPECT_OK(conversation->SendMessageAsync(user_messages,
                                           std::move(message_callbacks_2)));
  done_2.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(
      conversation->GetHistory(),
      testing::ElementsAre(user_message_1, assistant_message_1_for_confirm,
                           user_messages[0], user_messages[1],
                           assistant_message_2_for_confirm));
}

TEST_P(ConversationTest, SendMessageWithPreface) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message;
  if (prefill_preface_on_init_) {
    expected_message = {{"role", "assistant"},
                        {"content",
                         {{{"type", "text"},
                           {"text", " rupani rupani rupani echoes echoes"}}}}};
  } else {
    expected_message = {
        {"role", "assistant"},
        {"content",
         {{{"type", "text"},
           {"text", " noses</caption> গ্রাহ<unused5296> omp"}}}}};
  }
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  EXPECT_EQ(json_message, expected_message);
}

TEST_P(ConversationTest, GetBenchmarkInfo) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  proto::BenchmarkParams benchmark_params;
  engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  ASSERT_OK_AND_ASSIGN(const Message message_1,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_1,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_1.GetTotalPrefillTurns(),
            prefill_preface_on_init_ ? 2 : 1);

  ASSERT_OK_AND_ASSIGN(const Message message_2,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_2,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_2.GetTotalPrefillTurns(),
            prefill_preface_on_init_ ? 3 : 2);
}

TEST_P(ConversationTest, CancelGroupWithSendMessageAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  auto mock_task_controller1 = std::make_unique<MockTaskController>();
  // Expect Cancel() to be called on the first task controller when
  // CancelGroup("group1") is called.
  EXPECT_CALL(*mock_task_controller1, Cancel())
      .WillOnce(testing::Return(absl::OkStatus()));
  auto mock_task_controller2 = std::make_unique<MockTaskController>();
  // Expect Cancel() to be called on the second task controller when
  // CancelGroup("group1") is called.
  EXPECT_CALL(*mock_task_controller2, Cancel())
      .WillOnce(testing::Return(absl::OkStatus()));

  // Expect RunPrefillAsync to be called and return the first task controller.
  EXPECT_CALL(*mock_session_ptr, RunPrefillAsync(testing::_, testing::_))
      .WillOnce([&](const std::vector<InputData>& contents,
                    absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                        user_callback) {
        user_callback(Responses(TaskState::kDone));
        return std::move(mock_task_controller1);
      });
  // Expect RunDecodeAsync to be called and return the second task controller.
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [&](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
              const DecodeConfig& decode_config) {
            return std::move(mock_task_controller2);
          });

  absl::Notification done;
  absl::Status status;
  EXPECT_OK(
      conversation->SendMessageAsync(user_message,
                                     [&](absl::StatusOr<Message> message) {
                                       status = message.status();
                                       done.Notify();
                                     },
                                     {.task_group_id = "group1"}));

  conversation->CancelGroup("group1");
}

TEST_P(ConversationTest, CancelGroupWithRunTextScoringAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();

  auto cloned_session = std::make_unique<MockSession>();
  // Expect GetSessionConfig to be called on the cloned session.
  MockSession* cloned_session_ptr = cloned_session.get();
  EXPECT_CALL(*cloned_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config_));

  // Expect CloneAsync to be called and return the cloned session.
  EXPECT_CALL(*mock_session_ptr, CloneAsync(testing::_))
      .WillOnce(testing::Return(std::move(cloned_session)));
  auto mock_engine = CreateMockEngine(std::move(mock_session));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  auto mock_task_controller = std::make_unique<MockTaskController>();
  // Expect Cancel() to be called on the task controller when
  // CancelGroup("group1") is called.
  EXPECT_CALL(*mock_task_controller, Cancel())
      .WillOnce(testing::Return(absl::OkStatus()));

  // Expect RunTextScoringAsync to be called on the cloned session and return
  // the task controller.
  EXPECT_CALL(
      *cloned_session_ptr,
      RunTextScoringAsync(testing::ElementsAre("I am good."), testing::_, true))
      .WillOnce(
          [&](const std::vector<absl::string_view>& target_text,
              absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
              bool store_token_lengths) {
            return std::move(mock_task_controller);
          });

  absl::Notification done;
  std::string response_text;
  EXPECT_OK(conversation->RunTextScoringAsync(
      {"I am good."},
      [&](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        response_text = responses->GetTexts()[0];
        done.Notify();
      },
      {.task_group_id = "group1"}));

  conversation->CancelGroup("group1");
}

INSTANTIATE_TEST_SUITE_P(
    ConversationTest, ConversationTest,
    testing::ValuesIn(ConversationTest::GetTestParams()),
    [](const testing::TestParamInfo<ConversationTestParams>& info) {
      return absl::StrCat(
          info.param.enable_constrained_decoding ? "Constrained" : "Free", "_",
          info.param.prefill_preface_on_init ? "PrefillOnInit"
                                             : "NoPrefillOnInit");
    });

absl::AnyInvocable<void(absl::StatusOr<Message>)>
CreateCancelledMessageCallback(absl::Status& status, absl::Notification& done) {
  return [&status, &done](absl::StatusOr<Message> message) mutable {
    if (!message.ok()) {
      status = message.status();
      done.Notify();
      return;
    }
    if (auto json_message = std::get_if<JsonMessage>(&message.value());
        json_message->is_null()) {
      status = absl::OkStatus();
      done.Notify();
      return;
    }
    // Wait for a short time to slow down the decoding process, so that the
    // cancellation can be triggered in the middle of decoding.
    absl::SleepFor(absl::Milliseconds(100));
  };
}

TEST(ConversationAccessHistoryTest, AccessHistory) {
  // Create a Conversation.
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  // Send a message to the LLM.
  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  int partial_message_count = 0;
  absl::Notification done;
  EXPECT_OK(conversation->SendMessageAsync(
      user_message,
      CreateStreamingObserverCallback(partial_message_count, done)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));
  EXPECT_GT(partial_message_count, 0);

  // Get the history copy.
  auto history = conversation->GetHistory();
  ASSERT_THAT(history.size(), 2);
  ExpectAssistantMessageWithNonEmptyText(history.back());

  // Access the history with visitor function, and copy the last message.
  Message last_message;
  conversation->AccessHistory(
      [&last_message](const std::vector<Message>& history_view) {
        // Copy the last message to last_message. So we don't need to
        // copy the whole history, if we only need the last message.
        last_message = history_view.back();
      });
  EXPECT_THAT(last_message, testing::Eq(history.back()));
}

class ConversationCancellationTest : public testing::TestWithParam<bool> {
 protected:
  bool use_benchmark_info_ = GetParam();
};

TEST_P(ConversationCancellationTest, CancelProcessWithBenchmarkInfo) {
  bool use_benchmark_info = use_benchmark_info_;
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  // Set a large max num tokens to ensure the decoding is not finished before
  // cancellation.
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(20);
  if (use_benchmark_info) {
    proto::BenchmarkParams benchmark_params;
    engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  }
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  absl::Status status;
  absl::Notification done_1;
  conversation
      ->SendMessageAsync(
          JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
          CreateCancelledMessageCallback(status, done_1))
      .IgnoreError();
  // Wait for a short time to ensure the decoding has started.
  absl::SleepFor(absl::Milliseconds(100));
  conversation->CancelProcess();
  // Wait for the callback to be done.
  done_1.WaitForNotificationWithTimeout(absl::Seconds(10));
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kCancelled));

  // The history should be empty after cancellation.
  EXPECT_THAT(conversation->GetHistory().size(), 0);

  // Resend the message after cancellation, and it should succeed.
  status = absl::OkStatus();
  absl::Notification done_2;
  conversation
      ->SendMessageAsync(
          JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
          CreateCancelledMessageCallback(status, done_2))
      .IgnoreError();
  EXPECT_OK(status);
  // Wait for the callback to be done.
  done_2.WaitForNotificationWithTimeout(absl::Seconds(10));
  // Without cancellation, the history should have two messages, user and
  // assistant.
  auto history = conversation->GetHistory();
  ASSERT_EQ(history.size(), 2);
  EXPECT_THAT(history[0], testing::VariantWith<JsonMessage>(JsonMessage{
                              {"role", "user"}, {"content", "Hello world!"}}));
  // TODO(b/450903294) - Because the cancellation is not fully rollbacked, the
  // assistant message content depends on at which step the cancellation is
  // triggered, and that is non-deterministic. Here we only check the role is
  // assistant.
  EXPECT_THAT(std::holds_alternative<JsonMessage>(history[1]),
              testing::IsTrue());
  EXPECT_EQ(std::get<JsonMessage>(history[1])["role"], "assistant");

  conversation->CancelProcess();
  // No op after cancellation again.
  EXPECT_THAT(conversation->GetHistory().size(), 2);
}

INSTANTIATE_TEST_SUITE_P(ConversationCancellationTest,
                         ConversationCancellationTest, testing::Bool(),
                         testing::PrintToStringParamName());

class MockConstraint : public Constraint {
 public:
  class MockState : public State {
   public:
    ~MockState() override = default;
  };
  MOCK_METHOD(std::unique_ptr<State>, Start, (), (const, override));
  MOCK_METHOD(bool, IsEnded, (const State& state), (const, override));
  MOCK_METHOD(int, GetVocabularySize, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<State>>, ComputeNext,
              (const State& state, int token), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Bitmap>>, ComputeBitmap,
              (const State& state), (const, override));
};

TEST_P(ConversationTest, SendMessageWithConstraint) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation with ExternalConstraintConfig.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetConstraintProviderConfig(ExternalConstraintConfig())
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Create a mock constraint.
  auto mock_constraint = std::make_unique<MockConstraint>();
  Constraint* mock_constraint_ptr = mock_constraint.get();
  ExternalConstraintArg constraint_arg;
  constraint_arg.constraint = std::move(mock_constraint);

  // Send a message with the constraint.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));

  // Verify that the constraint is passed to RunDecode.
  EXPECT_CALL(*mock_session_ptr,
              RunDecode(testing::Property(&DecodeConfig::GetConstraint,
                                          mock_constraint_ptr)))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(
          user_message, {
                            .decoding_constraint = std::move(constraint_arg),
                        }));
}

TEST_P(ConversationTest, Clone) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Send a message to populate history.
  JsonMessage user_message = {{"role", "user"}, {"content", "Hello"}};
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(Responses(TaskState::kProcessing, {"Hi"})));
  ASSERT_OK(conversation->SendMessage(user_message));

  // Expect Session::Clone to be called.
  auto cloned_mock_session = std::make_unique<MockSession>();
  MockSession* cloned_mock_session_ptr = cloned_mock_session.get();
  EXPECT_CALL(*cloned_mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config_));
  EXPECT_CALL(*mock_session_ptr, Clone())
      .WillOnce(testing::Return(std::move(cloned_mock_session)));

  // Clone the conversation.
  ASSERT_OK_AND_ASSIGN(auto cloned_conversation, conversation->Clone());

  // Verify the history in the cloned conversation.
  auto history = cloned_conversation->GetHistory();
  EXPECT_EQ(history.size(), 2);
  EXPECT_EQ(std::get<JsonMessage>(history[0]), user_message);

  // Verify that sending a message in the cloned conversation works and uses the
  // cloned session.
  JsonMessage user_message2 = {{"role", "user"}, {"content", "How are you?"}};
  EXPECT_CALL(*cloned_mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*cloned_mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK(cloned_conversation->SendMessage(user_message2));

  // Verify that the original conversation is unaffected by the new message in
  // the cloned one.
  EXPECT_EQ(conversation->GetHistory().size(), 2);
  EXPECT_EQ(cloned_conversation->GetHistory().size(), 4);
}

TEST_P(ConversationTest, SendMessageWithMaxOutputTokens) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation with default config.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));

  // Verify that the max_output_tokens is passed to RunDecode.
  EXPECT_CALL(*mock_session_ptr,
              RunDecode(testing::Property(&DecodeConfig::GetMaxOutputTokens,
                                          std::make_optional(42))))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(user_message, {.max_output_tokens = 42}));
}

TEST(AppendMessageTest, Gemma3Sync) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  session_config.SetApplyPromptTemplateInSession(false);
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  ASSERT_OK_AND_ASSIGN(
      auto tokenizer,
      SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
              .string()));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  std::string template_text =
      ReadFile(GetTestdataPath(kGemma3ToolsMultiPrefillTemplatePath));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(template_text))
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Append the 1st message.
  absl::string_view expected_prefill_1 =
      "<start_of_turn>user\nYou are a helpful "
      "assistant.\n\n<end_of_turn>\n<start_of_turn>user\nHello world!";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_1)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
      {.has_pending_message = true}));

  // Append the 2nd message.
  absl::string_view expected_prefill_2 = " This is a long message.";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_2)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", " This is a long message."}},
      {.has_pending_message = true}));

  // Append the 3rd message.
  absl::string_view expected_prefill_3 = " continuing...";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_3)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", " continuing..."}},
      {.has_pending_message = true}));

  // Finish appending message.
  absl::string_view expected_prefill_4 =
      " The message is ended.<end_of_turn>\n<start_of_turn>model\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_4)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));
  ASSERT_OK_AND_ASSIGN(
      const Message response_appending,
      conversation->SendMessage(JsonMessage{
          {"role", "user"}, {"content", " The message is ended."}}));
}

TEST(AppendMessageTest, Gemma3Async) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  session_config.SetApplyPromptTemplateInSession(false);
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  ASSERT_OK_AND_ASSIGN(
      auto tokenizer,
      SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
              .string()));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  std::string template_text =
      ReadFile(GetTestdataPath(kGemma3ToolsMultiPrefillTemplatePath));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(template_text))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  auto test_callback =
      [](const std::vector<InputData>& contents,
         absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      };

  // Append the 1st message.
  absl::string_view expected_prefill_1 = "<start_of_turn>user\nHello world!";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_1))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done1;
  ASSERT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
      [&done1](absl::StatusOr<Message> message) { done1.Notify(); },
      {.has_pending_message = true}));
  done1.WaitForNotificationWithTimeout(absl::Seconds(3));

  // Append the 2nd message.
  absl::string_view expected_prefill_2 = " This is a long message.";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_2))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done2;
  ASSERT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", " This is a long message."}},
      [&done2](absl::StatusOr<Message> message) { done2.Notify(); },
      {.has_pending_message = true}));
  done2.WaitForNotificationWithTimeout(absl::Seconds(3));

  // Append the 3rd message.
  absl::string_view expected_prefill_3 = " continuing...";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_3))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done3;
  ASSERT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", " continuing..."}},
      [&done3](absl::StatusOr<Message> message) { done3.Notify(); },
      {.has_pending_message = true}));
  done3.WaitForNotificationWithTimeout(absl::Seconds(3));

  // Append the 4th message.
  absl::string_view expected_prefill_4 = " The message is ended.";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_4))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done4;
  EXPECT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", " The message is ended."}},
      [&done4](absl::StatusOr<Message> message) { done4.Notify(); },
      {.has_pending_message = true}));
  done4.WaitForNotificationWithTimeout(absl::Seconds(3));

  // The 5th message triggers the decode.
  absl::string_view expected_prefill_5 =
      "<end_of_turn>\n<start_of_turn>model\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_5))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });
  Message expected_assistant_message =
      JsonMessage({{"role", "assistant"},
                   {"content", {{{"type", "text"}, {"text", "I am good."}}}}});
  absl::Notification done5;
  // Trigger the decode by sending an empty message.
  EXPECT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", ""}},
      CreateTestMessageCallback(expected_assistant_message, done5),
      {.has_pending_message = false}));
  done5.WaitForNotificationWithTimeout(absl::Seconds(3));
}

TEST(AppendMessageTest, Gemma3SyncPrefillPrefaceOnInitAndAlternateRoles) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  session_config.SetApplyPromptTemplateInSession(false);
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  ASSERT_OK_AND_ASSIGN(
      auto tokenizer,
      SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
              .string()));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  std::string template_text =
      ReadFile(GetTestdataPath(kGemma3ToolsMultiPrefillTemplatePath));

  // Init with preface.
  absl::string_view expected_prefill_preface = R"(<start_of_turn>system
def tool_name(
    x: int | None = None,
) -> dict:
  """
  Args:
    x  """

<end_of_turn>
<start_of_turn>user
You are a helpful assistant.

<end_of_turn>
)";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(testing::VariantWith<InputText>(
                  testing::Property(&InputText::GetRawTextString,
                                    expected_prefill_preface)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(template_text))
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}},
              .tools = nlohmann::ordered_json::parse(
                  R"json([{
                            "name": "tool_name",
                            "parameters": {
                              "properties": {
                                "x": {
                                  "type": "integer"
                                }
                              }
                            }
                          }])json")})
          .SetPrefillPrefaceOnInit(true)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Append the 1st message.
  absl::string_view expected_prefill_1 = "<start_of_turn>user\nHello world!";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_1)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
      {.has_pending_message = true}));

  // Append the 2nd message.
  absl::string_view expected_prefill_2 =
      "<end_of_turn>\n<start_of_turn>model\nNice to meet you.";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_2)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "model"}, {"content", "Nice to meet you."}},
      {.has_pending_message = true}));

  // Append the 3rd message.
  absl::string_view expected_prefill_3 = " How can I help you today?";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_3)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "model"}, {"content", " How can I help you today?"}},
      {.has_pending_message = true}));

  // Append the 4th message.
  absl::string_view expected_prefill_4 = " The message is ended.";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_4)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      JsonMessage{{"role", "model"}, {"content", " The message is ended."}},
      {.has_pending_message = true}));

  // Append the 5th message.
  absl::string_view expected_prefill_5 = R"(<end_of_turn>
<start_of_turn>user
```tool_outputs
{"location": "Paris", "temperature": 20, "unit": "C", "weather": "Sunny"})";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_5)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(
      conversation->SendMessage(JsonMessage{{"role", "tool"},
                                            {"content",
                                             {
                                                 {"type", "tool_response"},
                                                 {"tool_response",
                                                  {
                                                      {"location", "Paris"},
                                                      {"temperature", 20},
                                                      {"unit", "C"},
                                                      {"weather", "Sunny"},
                                                  }},
                                             }}},
                                {.has_pending_message = true}));

  // Append the 6th message.
  absl::string_view expected_prefill_6 =
      R"({"location": "London", "temperature": 15, "unit": "C", "weather": "Cloudy"}
```<end_of_turn>
<start_of_turn>model
)";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_6)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));
  ASSERT_OK(
      conversation->SendMessage(JsonMessage{{"role", "tool"},
                                            {"content",
                                             {
                                                 {"type", "tool_response"},
                                                 {"tool_response",
                                                  {
                                                      {"location", "London"},
                                                      {"temperature", 15},
                                                      {"unit", "C"},
                                                      {"weather", "Cloudy"},
                                                  }},
                                             }}},
                                {.has_pending_message = false}));
}

}  // namespace
}  // namespace litert::lm
