#include "runtime/components/sampling_cpu_util.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "absl/random/random.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(SamplingCpuUtilTest, TopKIndicies_BatchSize1) {
  const std::vector<float> logits = {0.1, 0.5, 0.4, 0.2};
  auto indices =
      TopKIndicies(absl::MakeConstSpan(logits), /*k=*/2, /*batch_size=*/1);
  EXPECT_TRUE(indices.ok());
  EXPECT_THAT(*indices, UnorderedElementsAre(1, 2));
}

TEST(SamplingCpuUtilTest, TopKIndicies_BatchSize2) {
  const std::vector<float> logits = {0.1, 0.5, 0.4, 0.2};
  auto indices =
      TopKIndicies(absl::MakeConstSpan(logits), /*k=*/1, /*batch_size=*/2);
  EXPECT_TRUE(indices.ok());
  EXPECT_THAT(*indices, ElementsAre(1, 0));
}

TEST(SamplingCpuUtilTest, Softmax_BatchSize1) {
  const std::vector<float> logits = {0.1f, 0.1f};
  const std::vector<int> topk_indices = {0, 1};
  std::vector<float> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/1.0, /*batch_size=*/1, max_logit_values);
  EXPECT_TRUE(probabilities.ok());
  EXPECT_THAT(*probabilities, ElementsAre(0.5, 0.5));
  EXPECT_THAT(max_logit_values, ElementsAre(0.1f));
}

TEST(SamplingCpuUtilTest, Softmax_AllZeroLogits) {
  const std::vector<float> logits = {0.0f, 0.0f};
  const std::vector<int> topk_indices = {0, 1};
  std::vector<float> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/1.0, /*batch_size=*/1, max_logit_values);
  EXPECT_TRUE(probabilities.ok());
  EXPECT_THAT(*probabilities, ElementsAre(0.5, 0.5));
  EXPECT_THAT(max_logit_values, ElementsAre(0.0f));
}

TEST(SamplingCpuUtilTest, Softmax_TemperatureZero) {
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f};
  const std::vector<int> topk_indices = {0, 1, 2};
  std::vector<float> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/0.00000001f, /*batch_size=*/1, max_logit_values);
  EXPECT_TRUE(probabilities.ok());
  // Very small temperature should mimic greedy sampling.
  EXPECT_THAT(*probabilities, ElementsAre(0.0f, 0.0f, 1.0f));
  EXPECT_THAT(max_logit_values, ElementsAre(2.0f));
}

TEST(SamplingCpuUtilTest, Softmax_TemperatureInf) {
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
  const std::vector<int> topk_indices = {0, 1, 2, 3};
  std::vector<float> max_logit_values;
  auto probabilities = Softmax(
      absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
      /*temperature=*/100000000000.0f, /*batch_size=*/1, max_logit_values);
  EXPECT_TRUE(probabilities.ok());
  // Very large temperature should mimic uniform sampling.
  EXPECT_THAT(*probabilities, ElementsAre(0.25f, 0.25f, 0.25f, 0.25f));
  EXPECT_THAT(max_logit_values, ElementsAre(3.0f));
}

TEST(SamplingCpuUtilTest, Softmax_BatchSize3) {
  // Batch size of 3, vocab size of 2.
  const std::vector<float> logits = {0.1f, 0.1f, 0.0f, 5.0f, 1.0f, 0.0f};
  absl::Span<const float> logits_span = absl::MakeConstSpan(logits);
  const std::vector<int> topk_indices = {0, 1, 0, 1, 0, 1};
  absl::Span<const int> topk_indices_span = absl::MakeConstSpan(topk_indices);
  std::vector<float> max_logit_values;
  auto probabilities =
      Softmax(logits_span, topk_indices_span,
              /*temperature=*/1.0f, /*batch_size=*/3, max_logit_values);
  EXPECT_TRUE(probabilities.ok());
  EXPECT_THAT(*probabilities,
              ElementsAre(0.5f, 0.5f, 0.00669285096f, 0.993307173f,
                          0.731058598f, 0.268941432f));
  EXPECT_THAT(max_logit_values, ElementsAre(0.1f, 5.0f, 1.0f));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_InvalidInputs) {
  const std::vector<float> probabilities = {0.0, 0.0, 0.3};
  absl::BitGen rng;
  // Negative k.
  std::vector<float> sampled_scores;
  auto sampled_ids = TopKTopPSampling(
      absl::MakeConstSpan(probabilities), /*k=*/-1,
      /*p=*/0.5,
      /*temperature=*/1.0, rng, /*batch_size=*/1, sampled_scores);
  EXPECT_FALSE(sampled_ids.ok());
  // Negative p.
  sampled_ids = TopKTopPSampling(absl::MakeConstSpan(probabilities),
                                 /*k=*/1,
                                 /*p=*/-0.5, /*temperature=*/1.0f, rng,
                                 /*batch_size=*/1, sampled_scores);
  EXPECT_FALSE(sampled_ids.ok());
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize1) {
  const std::vector<float> probabilities = {0.0, 0.0, 0.3};
  absl::BitGen rng;
  std::vector<float> sampled_scores;
  auto sampled_ids = TopKTopPSampling(
      absl::MakeConstSpan(probabilities), /*k=*/1,
      /*p=*/0.5,
      /*temperature=*/1.0f, rng, /*batch_size=*/1, sampled_scores);
  EXPECT_TRUE(sampled_ids.ok());
  EXPECT_THAT((*sampled_ids), ElementsAre(2));
  EXPECT_THAT(sampled_scores, ElementsAre(1.0));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize2) {
  // Batch of 3, vocab size of 3. The sampled ids are 2, 1, 0.
  const std::vector<float> logits = {0.0, 0.0, 1.0, 0.0, 1.0,
                                            0.0, 1.0, 0.0, 0.0};
  absl::BitGen rng;
  std::vector<float> sampled_scores;
  auto sampled_ids = TopKTopPSampling(
      absl::MakeConstSpan(logits), /*k=*/2, /*p=*/0.5,
      /*temperature=*/0.00001f, rng, /*batch_size=*/3, sampled_scores);
  EXPECT_TRUE(sampled_ids.ok());
  EXPECT_THAT((*sampled_ids), ElementsAre(2, 1, 0));
  EXPECT_THAT(sampled_scores, ElementsAre(1.0, 1.0, 1.0));
}

}  // namespace
}  // namespace litert::lm
