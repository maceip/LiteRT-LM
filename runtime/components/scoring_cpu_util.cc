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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/sampling_cpu_util.h"

namespace litert::lm {

absl::StatusOr<std::vector<float>> ComputeBatchConfidences(
    absl::Span<const float> logits, const std::vector<int>& sampled_ids,
    float temperature) {
  int batch_size = sampled_ids.size();
  const int vocab_size = logits.size() / batch_size;
  // Get all indices and their probabilities for calculating perplexity.
  auto all_indices = TopKIndicies(logits, vocab_size, batch_size);
  if (!all_indices.ok()) return all_indices.status();
  std::vector<float> all_logit_values;
  auto all_probabilities =
      Softmax(logits, *all_indices, temperature, batch_size, all_logit_values);
  if (!all_probabilities.ok()) return all_probabilities.status();
  std::vector<float> batch_confidence = std::vector<float>(batch_size, 0.0f);
  for (int b = 0; b < batch_size; ++b) {
    if (sampled_ids[b] >= 0 && sampled_ids[b] < vocab_size) {
      int sampled_index = b * vocab_size + sampled_ids[b];
      batch_confidence[b] = -1 * std::log((*all_probabilities)[sampled_index]);
    } else if (sampled_ids[b] == -1) {
      // Special value for a batch stream that has ended.
      batch_confidence[b] = 0;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid sampled id: ", sampled_ids[b]));
    }
  }
  return batch_confidence;
}

}  // namespace litert::lm
