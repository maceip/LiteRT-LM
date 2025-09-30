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

#include "runtime/util/lora_data.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

std::string GetLoraFilePath() {
  auto path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_gpu_lora_rank32_f16_all_ones.tflite";
  return path.string();
}

TEST(LoraDataTest, CanCreateLoraDataFromScopedFile) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ::litert::lm::ModelAssets::Create(GetLoraFilePath()));
  ASSERT_OK_AND_ASSIGN(auto scoped_file, model_assets.GetOrCreateScopedFile());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora,
                       LoraData::CreateFromScopedFile(std::move(scoped_file)));
}

TEST(LoraDataTest, CanCreateLoraDataFromFilePath) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora,
                       LoraData::CreateFromFilePath(GetLoraFilePath()));
}

}  // namespace
}  // namespace litert::lm
