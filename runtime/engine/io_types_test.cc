#include "runtime/engine/io_types.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model.h"  // from @litert
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAreArray;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;
using ::testing::ContainsRegex;

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTestTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    BuildLayout(kTensorDimensions)};

std::string FloatToString(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

TEST(InputTextTest, GetText) {
  InputText input_text("Hello World!");
  EXPECT_EQ(input_text.GetData(), "Hello World!");
}

TEST(InputTextTest, ToString) {
  InputText input_text("Hello World!");
  EXPECT_EQ(ToString(input_text), "Hello World!");
}

TEST(InputImageTest, GetRawImageBytes) {
  InputImage input_image("Hello Image!");
  ASSERT_OK_AND_ASSIGN(auto raw_image_bytes, input_image.GetRawImageBytes());
  EXPECT_EQ(raw_image_bytes, "Hello Image!");
}

TEST(InputImageTest, GetPreprocessedImageTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env.Get(), kTensorBufferType, kTensorType,
                                  kTensorSize));

  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  // Create an InputImage from the tensor buffer. This InputImage takes
  // ownership of the tensor buffer.
  InputImage input_image(std::move(original_tensor_buffer));

  // Confirm the InputImage is preprocessed.
  EXPECT_TRUE(input_image.IsPreprocessed());

  // Confirm the retrieved tensor buffer is identical to the original tensor
  // buffer.
  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       input_image.GetPreprocessedImageTensor());

  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_size,
                              retrieved_tensor_buffer->Size());
  EXPECT_EQ(retrieved_tensor_buffer_size, kTensorSize);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_type,
                              retrieved_tensor_buffer->BufferType());
  EXPECT_EQ(retrieved_tensor_buffer_type, kTensorBufferType);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_type,
                              retrieved_tensor_buffer->TensorType());
  EXPECT_EQ(retrieved_tensor_type, kTensorType);

  // Confirm the value of the retrieved_tensor_buffer is identical to
  // kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputAudioTest, GetRawAudioBytes) {
  InputAudio input_audio("Hello Audio!");
  ASSERT_OK_AND_ASSIGN(auto raw_audio_bytes, input_audio.GetRawAudioBytes());
  EXPECT_EQ(raw_audio_bytes, "Hello Audio!");
}

TEST(InputAudioTest, GetPreprocessedAudioTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env.Get(), kTensorBufferType, kTensorType,
                                  kTensorSize));

  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  // Create an InputAudio from the tensor buffer. This InputAudio takes
  // ownership of the tensor buffer.
  InputAudio input_audio(std::move(original_tensor_buffer));

  // Confirm the InputAudio is preprocessed.
  EXPECT_TRUE(input_audio.IsPreprocessed());

  // Confirm the retrieved tensor buffer is identical to the original tensor
  // buffer.
  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       input_audio.GetPreprocessedAudioTensor());

  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_size,
                              retrieved_tensor_buffer->Size());
  EXPECT_EQ(retrieved_tensor_buffer_size, kTensorSize);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_type,
                              retrieved_tensor_buffer->BufferType());
  EXPECT_EQ(retrieved_tensor_buffer_type, kTensorBufferType);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_type,
                              retrieved_tensor_buffer->TensorType());
  EXPECT_EQ(retrieved_tensor_type, kTensorType);

  // Confirm the value of the retrieved_tensor_buffer is identical to
  // kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(ResponsesTest, GetResponseTextAt) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableResponseTexts()[0] = "Hello World!";
  responses.GetMutableResponseTexts()[1] = "How's it going?";
  EXPECT_THAT(responses.GetResponseTextAt(0), IsOkAndHolds("Hello World!"));
  EXPECT_THAT(responses.GetResponseTextAt(1), IsOkAndHolds("How's it going?"));
  EXPECT_THAT(responses.GetResponseTextAt(2),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, GetScoreAt) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
  EXPECT_THAT(responses.GetScoreAt(0), IsOkAndHolds(0.1));
  EXPECT_THAT(responses.GetScoreAt(1), IsOkAndHolds(0.2));
  EXPECT_THAT(responses.GetScoreAt(2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, GetMutableScores) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
  EXPECT_THAT(responses.GetScoreAt(0), IsOkAndHolds(0.1));
  EXPECT_THAT(responses.GetScoreAt(1), IsOkAndHolds(0.2));
  EXPECT_THAT(responses.GetScoreAt(2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResponsesTest, HasScores) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableScores()[0] = 0.1;
  responses.GetMutableScores()[1] = 0.2;
}

TEST(ResponsesTest, GetMutableResponseTexts) {
  Responses responses(/*num_output_candidates=*/2);
  responses.GetMutableResponseTexts()[0] = "Hello World!";
  responses.GetMutableResponseTexts()[1] = "How's it going?";
  EXPECT_THAT(responses.GetMutableResponseTexts()[0], "Hello World!");
  EXPECT_THAT(responses.GetMutableResponseTexts()[1], "How's it going?");
}

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndScores) {
  litert::lm::Responses responses(2);
  responses.GetMutableResponseTexts()[0] = "Hello";
  responses.GetMutableResponseTexts()[1] = "World";
  responses.GetMutableScores()[0] = 0.9f;
  responses.GetMutableScores()[1] = -0.5f;  // Test with a negative score

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Total candidates: 2:\n"
      "  Candidate 0 (score: " +
      FloatToString(0.9f) +
      "):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: " +
      FloatToString(-0.5f) +
      "):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndNoScores) {
  litert::lm::Responses responses(2);
  responses.GetMutableResponseTexts()[0] = "Hello";
  responses.GetMutableResponseTexts()[1] = "World";

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Total candidates: 2:\n"
      "  Candidate 0 (score: N/A):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: N/A):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

proto::BenchmarkParams GetBenchmarkParams() {
  proto::BenchmarkParams benchmark_params;
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  return benchmark_params;
}

// --- Test Init Phases ---
TEST(BenchmarkInfoTests, AddAndGetInitPhases) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeInitPhaseStart("Model Load"));
  EXPECT_OK(benchmark_info.TimeInitPhaseStart("Tokenizer Load"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeInitPhaseEnd("Tokenizer Load"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeInitPhaseEnd("Model Load"));

  const auto& phases = benchmark_info.GetInitPhases();
  ASSERT_EQ(phases.size(), 2);
  // The time should be greater than 50ms.
  EXPECT_GT(phases.at("Tokenizer Load"), absl::Milliseconds(50));
  // The time should be greater than 50 + 50 = 100ms.
  EXPECT_GT(phases.at("Model Load"), absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, AddInitPhaseTwice) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeInitPhaseStart("Model Load"));
  // Starting the same phase twice should fail.
  EXPECT_THAT(benchmark_info.TimeInitPhaseStart("Model Load"),
              StatusIs(absl::StatusCode::kInternal));

  // Ending a phase that has not started should fail.
  EXPECT_THAT(benchmark_info.TimeInitPhaseEnd("Tokenizer Load"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddPrefillTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(200));
  EXPECT_EQ(benchmark_info.GetTotalPrefillTurns(), 2);
}

TEST(BenchmarkInfoTests, AddPrefillTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  // Starting the prefill turn twice should fail.
  EXPECT_THAT(benchmark_info.TimePrefillTurnStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  // Ending a prefill turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimePrefillTurnEnd(200),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddDecodeTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(200));
  EXPECT_EQ(benchmark_info.GetTotalDecodeTurns(), 2);
}

TEST(BenchmarkInfoTests, AddDecodeTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  // Starting the decode turn twice should fail.
  EXPECT_THAT(benchmark_info.TimeDecodeTurnStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));
  // Ending a decode turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimeDecodeTurnEnd(200),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddMarks) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  EXPECT_EQ(benchmark_info.GetMarkDurations().size(), 1);

  // The time should record the duration between the 2nd and 3rd calls, which
  // should be slightly more than 200ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(200));
  // Verify that the time doesn't record the duration between the 1st and 3nd
  // calls, which is less than 200ms + 200ms = 400ms.
  EXPECT_LT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(400));
}

TEST(BenchmarkInfoTests, AddTwoMarks) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeMarkDelta("tokenize"));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeMarkDelta("tokenize"));
  EXPECT_EQ(benchmark_info.GetMarkDurations().size(), 2);

  // Time between two sampling calls should be more than 50ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(50));
  // Time between two tokenize calls should be more than 50ms + 50ms = 100ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("tokenize"),
            absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, OperatorOutputWithData) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeInitPhaseStart("Load Model"));
  EXPECT_OK(benchmark_info.TimeInitPhaseStart("Load Tokenizer"));
  EXPECT_OK(benchmark_info.TimeInitPhaseEnd("Load Model"));
  EXPECT_OK(benchmark_info.TimeInitPhaseEnd("Load Tokenizer"));

  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(200));

  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));

  std::stringstream ss;
  ss << benchmark_info;
  const std::string expected_output = R"(BenchmarkInfo:
  Init Phases \(2\):
    - Load Model: .* ms
    - Load Tokenizer: .* ms
    Total init time: .* ms
--------------------------------------------------
  Prefill Turns \(Total: 2\):
    Prefill Turn 1: Processed 100 tokens in .* duration.
      Prefill Speed: .* tokens/sec.
    Prefill Turn 2: Processed 200 tokens in .* duration.
      Prefill Speed: .* tokens/sec.
--------------------------------------------------
  Decode Turns \(Total: 1\):
    Decode Turn 1: Processed 100 tokens in .* duration.
      Decode Speed: .* tokens/sec.
--------------------------------------------------
)";
  EXPECT_THAT(ss.str(), ContainsRegex(expected_output));
}

}  // namespace
}  // namespace litert::lm
