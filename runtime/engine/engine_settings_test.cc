#include "runtime/engine/engine_settings.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::lm::EngineSettings;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::ContainsRegex;

std::shared_ptr<proto::LlmMetadata> CreateLlmMetadata() {
  proto::LlmMetadata llm_metadata;
  llm_metadata.mutable_start_token()->mutable_token_ids()->add_ids(2);
  llm_metadata.mutable_stop_tokens()->Add()->set_token_str("<eos>");
  llm_metadata.mutable_stop_tokens()->Add()->set_token_str("<end_of_turn>");
  llm_metadata.mutable_stop_tokens()->Add()->set_token_str("<ctrl>");
  llm_metadata.mutable_sampler_params()->set_type(
      proto::SamplerParameters::TOP_P);
  llm_metadata.mutable_sampler_params()->set_k(1);
  llm_metadata.mutable_sampler_params()->set_p(0.95f);
  llm_metadata.mutable_sampler_params()->set_temperature(1.0f);
  llm_metadata.mutable_sampler_params()->set_seed(0);
  return std::make_shared<proto::LlmMetadata>(llm_metadata);
}

TEST(EngineSettingsTest, GetModelPath) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  EXPECT_OK(settings);

  auto model_path =
      settings->GetMainExecutorSettings().GetModelAssets().GetPath();
  ASSERT_OK(model_path);
  EXPECT_EQ(*model_path, "test_model_path_1");
}

TEST(EngineSettingsTest, SetAndGetCacheDir) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  EXPECT_OK(settings);
  settings->GetMutableMainExecutorSettings().SetCacheDir("test_cache_dir");
  EXPECT_EQ(settings->GetMainExecutorSettings().GetCacheDir(),
            "test_cache_dir");
}

TEST(EngineSettingsTest, SetAndGetMaxNumTokens) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);

  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  EXPECT_OK(settings);
  settings->GetMutableMainExecutorSettings().SetMaxNumTokens(128);
  EXPECT_EQ(settings->GetMainExecutorSettings().GetMaxNumTokens(), 128);
}

TEST(EngineSettingsTest, SetAndGetExecutorBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);

  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::GPU);
  EXPECT_OK(settings);
  settings->GetMutableMainExecutorSettings().SetBackend(Backend::GPU);
  EXPECT_THAT(settings->GetMainExecutorSettings().GetBackend(),
              Eq(Backend::GPU));
}

TEST(EngineSettingsTest, DefaultExecutorBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  EXPECT_THAT(settings->GetMainExecutorSettings().GetBackend(),
              Eq(Backend::CPU));
}

TEST(EngineSettingsTest, BenchmarkParams) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  EXPECT_FALSE(settings->IsBenchmarkEnabled());

  proto::BenchmarkParams& benchmark_params =
      settings->GetMutableBenchmarkParams();
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  EXPECT_TRUE(settings->IsBenchmarkEnabled());
  EXPECT_EQ(settings->GetBenchmarkParams()->num_decode_tokens(), 100);
  EXPECT_EQ(settings->GetBenchmarkParams()->num_prefill_tokens(), 100);
}

TEST(EngineSettingsTest, LlmMetadata) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  EXPECT_FALSE(settings->GetLlmMetadata().has_value());

  proto::LlmMetadata& llm_metadata = settings->GetMutableLlmMetadata();
  llm_metadata.mutable_start_token()->set_token_str("test_token_str");
  EXPECT_TRUE(settings->GetLlmMetadata().has_value());
  EXPECT_EQ(settings->GetLlmMetadata().value().start_token().token_str(),
            "test_token_str");
}

class FakeTokenizer : public Tokenizer {
 public:
  FakeTokenizer() = default;

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    return std::vector<int>{1};
  }

  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override {
    return "fake_text";
  }

  absl::StatusOr<int> BosId() const override { return 2; }

  absl::StatusOr<int> EosId() const override { return 1; }
};

absl::Status IsExpectedLlmMetadata(const proto::LlmMetadata& llm_metadata) {
  if (!llm_metadata.has_start_token() ||
      llm_metadata.start_token().token_ids().ids_size() != 1 ||
      llm_metadata.start_token().token_ids().ids(0) != 2) {
    return absl::InvalidArgumentError("Start token is not set correctly.");
  }
  if (llm_metadata.stop_tokens_size() != 3) {
    return absl::InvalidArgumentError("Stop tokens size is not 3.");
  }
  if (llm_metadata.stop_tokens(0).token_ids().ids_size() != 1 ||
      llm_metadata.stop_tokens(0).token_ids().ids(0) != 1) {
    return absl::InvalidArgumentError("Stop tokens 0 is not set correctly.");
  }
  if (llm_metadata.stop_tokens(1).token_ids().ids_size() != 1 ||
      llm_metadata.stop_tokens(1).token_ids().ids(0) != 1) {
    return absl::InvalidArgumentError("Stop tokens 1 is not set correctly.");
  }
  if (llm_metadata.stop_tokens(2).token_ids().ids_size() != 1 ||
      llm_metadata.stop_tokens(2).token_ids().ids(0) != 1) {
    return absl::InvalidArgumentError("Stop tokens 2 is not set correctly.");
  }
  if (!llm_metadata.has_sampler_params() ||
      llm_metadata.sampler_params().type() != proto::SamplerParameters::TOP_P ||
      llm_metadata.sampler_params().k() != 1 ||
      llm_metadata.sampler_params().p() != 0.95f ||
      llm_metadata.sampler_params().temperature() != 1.0f ||
      llm_metadata.sampler_params().seed() != 0) {
    return absl::InvalidArgumentError("Sampler params is not set correctly.");
  }
  return absl::OkStatus();
}

TEST(EngineSettingsTest, MaybeUpdateAndValidate) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  std::shared_ptr<Tokenizer> tokenizer = std::make_shared<FakeTokenizer>();
  std::shared_ptr<proto::LlmMetadata> llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, llm_metadata));
  EXPECT_OK(IsExpectedLlmMetadata(settings->GetLlmMetadata().value()));
}

TEST(EngineSettingsTest, MaybeUpdateAndValidateQNN) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::QNN);
  EXPECT_OK(settings);

  std::shared_ptr<Tokenizer> tokenizer = std::make_shared<FakeTokenizer>();
  std::shared_ptr<proto::LlmMetadata> llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, llm_metadata));
  EXPECT_EQ(settings->GetLlmMetadata().value().sampler_params().type(),
            proto::SamplerParameters::TOP_P);
}

TEST(EngineSettingsTest, PrintOperator) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  proto::LlmMetadata& llm_metadata = settings->GetMutableLlmMetadata();
  llm_metadata.mutable_start_token()->set_token_str("test_token_str");
  proto::BenchmarkParams& benchmark_params =
      settings->GetMutableBenchmarkParams();
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  std::stringstream oss;
  oss << *settings;
}

TEST(SessionConfigTest, CreateDefault) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TYPE_UNSPECIFIED);
}

TEST(SessionConfigTest, SetAndGetSamplerParams) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  proto::SamplerParameters& sampler_params =
      session_config.GetMutableSamplerParams();
  sampler_params.set_type(proto::SamplerParameters::TOP_K);
  sampler_params.set_k(10);
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TOP_K);
  EXPECT_EQ(session_config.GetSamplerParams().k(), 10);

  // Mutable sampler params.
  session_config.GetMutableSamplerParams().set_type(
      proto::SamplerParameters::TYPE_UNSPECIFIED);
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TYPE_UNSPECIFIED);
}

TEST(SessionConfigTest, SetAndGetStopTokenIds) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableStopTokenIds() = {{0}, {1, 2}};
  EXPECT_EQ(session_config.GetStopTokenIds().size(), 2);
  EXPECT_THAT(session_config.GetStopTokenIds()[0], ElementsAre(0));
  EXPECT_THAT(session_config.GetStopTokenIds()[1], ElementsAre(1, 2));
}

TEST(SessionConfigTest, SetAndGetNumOutputCandidates) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetNumOutputCandidates(), 1);
  session_config.SetNumOutputCandidates(2);
  EXPECT_EQ(session_config.GetNumOutputCandidates(), 2);
}

TEST(SessionConfigTest, SetAndGetStartTokenId) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetStartTokenId(), -1);
  session_config.SetStartTokenId(1);
  EXPECT_EQ(session_config.GetStartTokenId(), 1);
}

TEST(SessionConfigTest, MaybeUpdateAndValidate) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  // We didn't call MaybeUpdateAndValidate on EngineSettings, so some of the
  // required fields are not set. So the validation should fail.
  EXPECT_THAT(session_config.MaybeUpdateAndValidate(*settings),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));

  std::shared_ptr<Tokenizer> tokenizer = std::make_shared<FakeTokenizer>();
  std::shared_ptr<proto::LlmMetadata> llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, llm_metadata));
  // The validation should pass now.
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
}

TEST(SessionConfigTest, PrintOperator) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams().set_type(
      proto::SamplerParameters::TOP_K);
  session_config.GetMutableSamplerParams().set_k(10);
  session_config.SetStartTokenId(1);
  session_config.GetMutableStopTokenIds() = {{0}, {1, 2}};
  session_config.SetNumOutputCandidates(2);
  std::stringstream oss;
  oss << session_config;
}

TEST(SessionConfigTest, SetAndGetSamplerBackend) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::CPU);
  session_config.SetSamplerBackend(Backend::GPU);
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::GPU);
}

}  // namespace
}  // namespace litert::lm
