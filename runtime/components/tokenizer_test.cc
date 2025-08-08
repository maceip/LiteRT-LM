#include "runtime/components/tokenizer.h"

#include <fcntl.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
};

TEST(TokenizerTest, TextToTensorBuffer) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TextToTokenIds("Hello World!"))
      .WillOnce(
          testing::Return(std::vector<int>{90, 547, 58, 735, 210, 466, 2294}));

  absl::string_view text = "Hello World!";
  auto ids_or = tokenizer->TextToTokenIds(text);
  EXPECT_TRUE(ids_or.ok());

  auto tensor_or = tokenizer->TokenIdsToTensorBuffer(ids_or.value());
  auto tensor = std::move(tensor_or.value());
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, tensor.TensorType());
  EXPECT_EQ(tensor_type.Layout().Dimensions(), ::litert::Dimensions({1, 7}));

  auto copied_data = CopyFromTensorBuffer2D<int>(tensor);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_THAT((*copied_data)[0],
              ::testing::ElementsAre(90, 547, 58, 735, 210, 466, 2294));
}

TEST(TokenizerTest, TensorBufferToTokenIds) {
  auto tokenizer = std::make_unique<MockTokenizer>();

  const std::vector<int> ids = {90,  547, 58, 735, 210, 466, 2294,
                                224, 24,  8,  66,  246, 18,  2295};
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer tensor_buffer,
                              CopyToTensorBuffer<int>(ids, {2, 7}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer_type,
                              tensor_buffer.TensorType());
  EXPECT_EQ(tensor_buffer_type.Layout().Dimensions(),
            ::litert::Dimensions({2, 7}));

  auto token_ids = Tokenizer::TensorBufferToTokenIds(tensor_buffer);
  EXPECT_TRUE(token_ids.ok());
  EXPECT_EQ(token_ids.value().size(), 2);
  EXPECT_EQ(token_ids.value()[0],
            std::vector<int>({90, 547, 58, 735, 210, 466, 2294}));
  EXPECT_EQ(token_ids.value()[1],
            std::vector<int>({224, 24, 8, 66, 246, 18, 2295}));
}

TEST(TokenizerTest, TokenIdsToTexts) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TokenIdsToText(::testing::_))
      .WillOnce(testing::Return("▁Hello▁World!"))
      .WillOnce(testing::Return("▁How's▁it▁going?"));

  const std::vector<std::vector<int>> ids = {{90, 547, 58, 735, 210, 466, 2294},
                                             {224, 24, 8, 66, 246, 18, 2295}};

  auto texts = tokenizer->TokenIdsToTexts(/*batch_size=*/2, ids);
  EXPECT_TRUE(texts.ok());
  EXPECT_EQ(texts.value().size(), 2);
  EXPECT_EQ(texts.value()[0].value(), "▁Hello▁World!");
  EXPECT_EQ(texts.value()[1].value(), "▁How's▁it▁going?");
}

TEST(TokenizerTest, TokenIdsToTextsWithIncompleteBPESequence) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TokenIdsToText(::testing::_))
      .WillOnce(testing::Return(absl::DataLossError("Incomplete BPE sequence")))
      .WillOnce(testing::Return("▁How's▁it▁going?"));

  const std::vector<std::vector<int>> ids = {{90, 547, 58, 735, 210, 466, 2294},
                                             {224, 24, 8, 66, 246, 18, 2295}};

  auto texts = tokenizer->TokenIdsToTexts(/*batch_size=*/2, ids);
  EXPECT_TRUE(texts.ok());
  EXPECT_EQ(texts.value().size(), 2);
  EXPECT_EQ(texts.value()[0].status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(texts.value()[1].value(), "▁How's▁it▁going?");
}

TEST(TokenizerTest, MergeTokenIds) {
  const std::vector<std::vector<int>> previous_ids = {{90, 547, 58, 735},
                                                      {224, 24}};
  const std::vector<std::vector<int>> current_ids = {{210, 466, 2294},
                                                     {8, 66, 246, 18, 2295}};
  auto merged = Tokenizer::MergeTokenIds(previous_ids, current_ids);
  EXPECT_TRUE(merged.ok());
  EXPECT_EQ(merged->size(), 2);
  EXPECT_EQ((*merged)[0], std::vector<int>({90, 547, 58, 735, 210, 466, 2294}));
  EXPECT_EQ((*merged)[1], std::vector<int>({224, 24, 8, 66, 246, 18, 2295}));
}

TEST(SentencePieceTokenizerTest, BosId) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_EQ(tokenizer->BosId().status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(SentencePieceTokenizerTest, EosId) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_EQ(tokenizer->EosId().status().code(),
            absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace litert::lm
