#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SCORING_CPU_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SCORING_CPU_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
// Calculates the confidence of the batch given the logits and the sampled ids.
// Summing the confidence of all batches will give the total perplexity.
// The logits are expected to be in the shape of [batch_size, vocab_size].
// The sampled_ids are the sampled token ids for the full batch.
// The temperature is used for calculating the softmax function.
// Returns the confidence i.e. negative log probability for the entire batch.
// Ranges from [0, inf)
absl::StatusOr<std::vector<float>> ComputeBatchConfidences(
    absl::Span<const float> logits, const std::vector<int>& sampled_ids,
    float temperature);
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SCORING_CPU_UTIL_H_
