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

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;

std::string GetLoraFilePath() {
  auto path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_gpu_lora_rank32_f16_all_ones.tflite";
  return path.string();
}

enum class LoraLoadType {
  kFilePath,
  kScopedFile,
  kBuffer,
};

class LoraDataTest : public ::testing::TestWithParam<LoraLoadType> {
 protected:
  absl::StatusOr<std::unique_ptr<LoraData>> CreateLoraData() {
    const LoraLoadType load_type = GetParam();
    switch (load_type) {
      case LoraLoadType::kFilePath: {
        return LoraData::CreateFromFilePath(GetLoraFilePath());
      }
      case LoraLoadType::kScopedFile: {
        ASSIGN_OR_RETURN(auto model_assets,
                         ::litert::lm::ModelAssets::Create(GetLoraFilePath()));
        ASSIGN_OR_RETURN(auto scoped_file,
                         model_assets.GetOrCreateScopedFile());
        return LoraData::CreateFromScopedFile(std::move(scoped_file));
      }
      case LoraLoadType::kBuffer: {
        ASSIGN_OR_RETURN(auto model_assets,
                         ::litert::lm::ModelAssets::Create(GetLoraFilePath()));
        ASSIGN_OR_RETURN(scoped_file_, model_assets.GetOrCreateScopedFile());
        ASSIGN_OR_RETURN(mapped_file_, ::litert::lm::MemoryMappedFile::Create(
                                           scoped_file_->file()));
        return LoraData::CreateFromBuffer(
            BufferRef<uint8_t>(mapped_file_->data(), mapped_file_->length()));
      }
    }
  }

 private:
  std::shared_ptr<const ScopedFile> scoped_file_;
  std::unique_ptr<MemoryMappedFile> mapped_file_;
};

TEST_P(LoraDataTest, CanCreateLoraData) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());
  EXPECT_NE(lora, nullptr);
}

TEST_P(LoraDataTest, GetLoraRank) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LoraData> lora, CreateLoraData());
  EXPECT_THAT(lora->GetLoRARank(), IsOkAndHolds(32));
}

INSTANTIATE_TEST_SUITE_P(
    LoraDataTests, LoraDataTest,
    ::testing::Values(LoraLoadType::kFilePath, LoraLoadType::kScopedFile,
                      LoraLoadType::kBuffer),
    [](const ::testing::TestParamInfo<LoraDataTest::ParamType>& info) {
      switch (info.param) {
        case LoraLoadType::kFilePath:
          return "FilePath";
        case LoraLoadType::kScopedFile:
          return "ScopedFile";
        case LoraLoadType::kBuffer:
          return "Buffer";
      }
    });

}  // namespace
}  // namespace litert::lm
