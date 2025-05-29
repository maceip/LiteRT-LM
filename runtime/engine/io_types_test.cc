#include "runtime/engine/io_types.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;
using ::testing::ContainsRegex;

std::string FloatToString(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
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
