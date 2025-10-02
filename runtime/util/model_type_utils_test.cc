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

#include "runtime/util/model_type_utils.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::NiceMock;
using ::testing::Return;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, BosId, (), (const, override));
  MOCK_METHOD(absl::StatusOr<int>, EosId, (), (const, override));
};

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3N) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(std::vector<int>{256000}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), tokenizer));
  EXPECT_THAT(model_type.has_gemma3n(), true);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3NWrongAudioToken) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(
          // The encoded ids for "<start_of_audio>" in the Gemma3 1B tokenizer.
          std::vector<int>{256001}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), tokenizer));
  EXPECT_THAT(model_type.has_gemma3n(), false);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(
          // The encoded ids for "<start_of_audio>" in the Gemma3 1B tokenizer.
          std::vector<int>{236820, 3041, 236779, 1340, 236779, 20156, 236813}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), tokenizer));
  EXPECT_THAT(model_type.has_gemma3(), true);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGenericModel) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("Hello"));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), tokenizer));
  EXPECT_THAT(model_type.has_generic_model(), true);
}

}  // namespace
}  // namespace litert::lm
