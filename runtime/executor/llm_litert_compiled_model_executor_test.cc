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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {
const int kMaxNumTokens = 32;
const int kNumThreads = 4;

using ::litert::lm::Backend;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::LlmLiteRtCompiledModelExecutor;
using ::litert::lm::ModelAssetBundleResources;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelResourcesTask;

absl::StatusOr<std::unique_ptr<ModelResources>> CreateExecutorModelResources(
    absl::string_view model_path) {
  auto scoped_file = ScopedFile::Open(model_path);
  auto resources = ModelAssetBundleResources::Create(
      /*tag=*/"", std::move(*scoped_file));
  auto model_resources = ModelResourcesTask::Create(std::move(*resources));
  return model_resources;
}

TEST(LlmLiteRTCompiledModelExecutorTest, CreateExecutorTest_WithoutCache) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto executor = LlmLiteRtCompiledModelExecutor::Create(*executor_settings,
                                                         env, *model_resources);
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

TEST(LlmLiteRTCompiledModelExecutorTest, PrefillTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutor::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);


  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());

  EXPECT_EQ(current_step, 3);
}

TEST(LlmLiteRTCompiledModelExecutorTest, DecodeTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutor::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);


  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tokens,
                              CreateTensorBuffer<int>({1, 1}));

  {
    EXPECT_OK(executor->Decode(output_tokens));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 4);

    auto output_tokens_span = ReferTensorBufferAsSpan<int>(output_tokens);
    EXPECT_EQ((*output_tokens_span)[0], 8005);
  }

  {
    EXPECT_OK(executor->Decode(output_tokens));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 5);

    auto output_tokens_span = ReferTensorBufferAsSpan<int>(output_tokens);
    EXPECT_EQ((*output_tokens_span)[0], 52530);
  }
}

TEST(LlmLiteRTCompiledModelExecutorTest, ConstrainedDecodeTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutor::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);


  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tokens,
                              CreateTensorBuffer<int>({1, 1}));


  ExecutorDecodeParams params;

  auto constraint = FakeConstraint({1, 2, 3}, /*vocabulary_size=*/262144);
  ConstrainedDecoder constraint_decoder =
      ConstrainedDecoder(&constraint, /*batch_size=*/1);
  params.SetConstraintDecoder(&constraint_decoder);

  {
    EXPECT_OK(executor->Decode(output_tokens, params));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 4);

    auto output_tokens_span = ReferTensorBufferAsSpan<int>(output_tokens);
    EXPECT_EQ((*output_tokens_span)[0], 2);
  }

  {
    EXPECT_OK(executor->Decode(output_tokens, params));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 5);

    auto output_tokens_span = ReferTensorBufferAsSpan<int>(output_tokens);
    EXPECT_EQ((*output_tokens_span)[0], 3);
  }
}

TEST(LlmLiteRTCompiledModelExecutorTest, DecodeLogitsTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutor::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);


  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 1);
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tokens,
                              CreateTensorBuffer<int>({1, 1}));

  {
    ASSERT_OK_AND_ASSIGN(auto output_logits, executor->DecodeLogits(inputs));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 1);

    auto output_logits_span = ReferTensorBufferAsSpan<float>(output_logits);
    EXPECT_TRUE(output_logits_span.HasValue());
  }
}

TEST(LlmLiteRTCompiledModelExecutorTest, CreateExecutorTest_WithCache) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
                    absl::StrCat("cache-", std::rand());
  std::filesystem::remove_all(cache_path);
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };

  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetCacheDir(cache_path.string());
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto executor = LlmLiteRtCompiledModelExecutor::Create(*executor_settings,
                                                         env, *model_resources);
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

TEST(LlmLiteRTCompiledModelExecutorTest,
     CreateExecutorTest_WithFileDescriptorCache) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
                    absl::StrCat("cache-", std::rand(), ".cache");
  std::filesystem::remove_all(cache_path);
  {
    // Create an empty file - ScopedFile expects the file to exist.
    std::ofstream cache_file(cache_path.string());
  }
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };
  ASSERT_OK_AND_ASSIGN(auto scoped_cache_file,
                       ScopedFile::OpenWritable(cache_path.string()));
  auto shared_scoped_cache_file =
      std::make_shared<ScopedFile>(std::move(scoped_cache_file));

  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(*model_assets, Backend::CPU);
  executor_settings->SetScopedCacheFile(shared_scoped_cache_file);
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto executor = LlmLiteRtCompiledModelExecutor::Create(*executor_settings,
                                                         env, *model_resources);
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

}  // namespace
}  // namespace litert::lm
