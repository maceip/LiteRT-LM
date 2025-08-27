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

#include "runtime/components/scoring_cpu_util.h"

#include <cmath>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
namespace {

TEST(ScoringCpuUtilTest, ComputeBatchConfidences_InvalidSampledId) {
  std::vector<float> logits = {0.0, 0.0, 0.3};
  std::vector<int> sampled_ids = {12};
  auto batchconfidence =
      ComputeBatchConfidences(absl::MakeConstSpan(logits), sampled_ids,
                         /*temperature=*/1.0);
  EXPECT_FALSE(batchconfidence.ok());
}

TEST(ScoringCpuUtilTest, ComputeBatchConfidences_BatchSize1) {
  std::vector<float> logits = {0.0, 0.0, 0.3};
  std::vector<int> sampled_ids = {2};
  auto batchconfidence =
      ComputeBatchConfidences(absl::MakeConstSpan(logits), sampled_ids,
                         /*temperature=*/1.0);
  EXPECT_TRUE(batchconfidence.ok());
  EXPECT_THAT(*batchconfidence,
              testing::ElementsAre(
                  testing::FloatNear(
                      -1 * std::log(exp(0.3f) / (2 + std::exp(0.3f))), 1e-6f)));
}

TEST(ScoringCpuUtilTest, ComputeBatchConfidences_BatchSize2) {
  std::vector<float> logits = {0.0, 0.0, 0.3, 0.0, 0.7, 0.0};
  std::vector<int> sampled_ids = {2, 1};
  auto batchconfidence = ComputeBatchConfidences(absl::MakeConstSpan(logits),
                                                sampled_ids,
                                                /*temperature=*/1.0);
  EXPECT_TRUE(batchconfidence.ok());
  EXPECT_THAT(*batchconfidence,
              testing::ElementsAre(
                  testing::FloatNear(
                      -1 * std::log(exp(0.3f) / (2 + std::exp(0.3f))), 1e-6f),
                  testing::FloatNear(
                      -1 * std::log(exp(0.7f) / (2 + std::exp(0.7f))), 1e-6f)));
}

TEST(ScoringCpuUtilTest, ComputeBatchConfidences_BatchSize2_OneStreamEnded) {
  std::vector<float> logits = {0.0, 0.0, 0.3, 0.0, 0.7, 0.0};
  std::vector<int> sampled_ids = {2, -1};
  auto batchconfidence = ComputeBatchConfidences(absl::MakeConstSpan(logits),
                                                 sampled_ids,
                                                  /*temperature=*/1.0);
  EXPECT_TRUE(batchconfidence.ok());
  EXPECT_THAT(*batchconfidence,
              testing::ElementsAre(
                  testing::FloatNear(
                      -1 * std::log(exp(0.3f) / (2 + std::exp(0.3f))), 1e-6f),
                  testing::FloatNear(0.0f, 1e-6f)));
}

}  // namespace
}  // namespace litert::lm
