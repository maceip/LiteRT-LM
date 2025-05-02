#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_FACTORY_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Creates a Sampler instance based on the provided parameters.
//
// Args:
//   backend: The backend implementation of the sampler (CPU / GPU / ...).
//   batch_size: The batch size for the input logits.
//   sampler_params: The parameters for the sampler.
//   The following parameters are optional and only used for GPU backend.
//   env: The litert environment to use for the sampler.
//   cache_size: The cache size for the sampler.
//   vocab_size: The vocabulary size for the sampler.
//   activation_data_type: The activation data type for the sampler.
//
// Returns:
//   The created Sampler instance.
absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    Environment* env = nullptr, std::optional<int> cache_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_FACTORY_H_
