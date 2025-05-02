#include "runtime/components/sampler_factory.h"

#include <memory>
#include <optional>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {
namespace {
absl::StatusOr<std::unique_ptr<Sampler>> CreateCpuSampler(
    int batch_size, proto::SamplerParameters sampler_params) {
  switch (sampler_params.type()) {
    case proto::SamplerParameters::TYPE_UNSPECIFIED:
      ABSL_LOG(INFO) << "Sampler type is unspecified. Assume the LLM Executor "
                        "handles the sampling logic.";
      return nullptr;
    case proto::SamplerParameters::TOP_P:
      return TopPSampler::Create(sampler_params.k(), sampler_params.p(),
                                 sampler_params.temperature(), batch_size,
                                 sampler_params.seed());
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Sampler type: ", sampler_params.type(), " not implemented yet."));
  }
}
}  // namespace

absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    Environment* env, std::optional<int> cache_size,
    std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  switch (backend) {
    case Backend::CPU:
      return CreateCpuSampler(batch_size, sampler_params);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ", backend));
  }
}
}  // namespace litert::lm
