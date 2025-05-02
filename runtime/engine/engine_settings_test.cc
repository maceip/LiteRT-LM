#include "runtime/engine/engine_settings.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"

namespace litert::lm {
namespace {

using ::litert::lm::EngineSettings;
using ::litert::lm::LlmExecutorSettings;
using ::testing::Eq;

TEST(EngineSettingsTest, GetModelPath) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorSettings executor_settings(model_assets);
  EngineSettings settings(executor_settings);

  EXPECT_EQ(settings.GetMainExecutorSettings().GetModelAssets().model_paths[0],
            "test_model_path_1");
}

TEST(EngineSettingsTest, SetAndGetCacheDir) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir("test_cache_dir");
  EngineSettings settings(executor_settings);
  EXPECT_EQ(settings.GetMainExecutorSettings().GetCacheDir(), "test_cache_dir");
}

TEST(EngineSettingsTest, SetAndGetMaxNumTokens) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorSettings executor_settings(model_assets);
  executor_settings.SetMaxNumTokens(128);
  EngineSettings settings(executor_settings);
  EXPECT_EQ(settings.GetMainExecutorSettings().GetMaxNumTokens(), 128);
}

TEST(EngineSettingsTest, SetAndGetExecutorBackend) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorSettings executor_settings(model_assets);
  executor_settings.SetBackend(Backend::GPU);
  EngineSettings settings(executor_settings);
  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::GPU));
}

TEST(EngineSettingsTest, DefaultExecutorBackend) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorSettings executor_settings(model_assets);
  EngineSettings settings(executor_settings);
  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::CPU));
}

TEST(EngineSettingsTest, BenchmarkParams) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorSettings executor_settings(model_assets);
  EngineSettings settings(executor_settings);
  EXPECT_FALSE(settings.IsBenchmarkEnabled());

  proto::BenchmarkParams benchmark_params;
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  settings.SetBenchmarkParams(benchmark_params);
  EXPECT_TRUE(settings.IsBenchmarkEnabled());
  EXPECT_EQ(settings.GetBenchmarkParams()->num_decode_tokens(), 100);
  EXPECT_EQ(settings.GetBenchmarkParams()->num_prefill_tokens(), 100);
}

TEST(SessionConfigTest, CreateDefault) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TOP_P);
  EXPECT_EQ(session_config.GetSamplerParams().k(), 1);
  EXPECT_EQ(session_config.GetSamplerParams().p(), 0.95f);
  EXPECT_EQ(session_config.GetSamplerParams().temperature(), 1.0f);
  EXPECT_EQ(session_config.GetSamplerParams().seed(), 0);
}

}  // namespace
}  // namespace litert::lm
