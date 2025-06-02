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

#include "runtime/executor/litert_compiled_model_executor_utils.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::_;  // NOLINT: Required by ASSERT_OK_AND_ASSIGN().

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromPath) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";

  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromScopedFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));

  auto model_assets =
      ModelAssets::Create(std::make_shared<ScopedFile>(std::move(model_file)));
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

}  // namespace
}  // namespace litert::lm
