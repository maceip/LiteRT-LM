#include "runtime/components/sampler_factory.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_shared_library.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

absl::Status CreateStatus(int error_code, const char* error_msg) {
  absl::StatusCode code = static_cast<absl::StatusCode>(error_code);
  return absl::Status(code, error_msg);
}

absl::Status CreateStatusAndFreeErrorMsg(int error_code, char* error_msg) {
  absl::Cleanup cleanup = [error_msg] { free(error_msg); };
  return error_code == 0 ? absl::OkStatus()
                         : CreateStatus(error_code, error_msg);
}

// A wrapper of TopKOpenClSampler C API functions that handles the lifetime of
// the resources.
class TopKOpenClCApiSampler : public Sampler {
 public:
  static absl::StatusOr<std::unique_ptr<TopKOpenClCApiSampler>> Create(
      LiteRtEnvironment env, int batch_size, int vocab_size,
      std::optional<ActivationDataType> activation_data_type,
      proto::SamplerParameters sampler_params) {
    // Load Sampler C API library and get the symbols.
    ASSIGN_OR_RETURN(std::unique_ptr<TopKOpenClSamplerCApi> capi,
                     GetTopKOpenClSamplerCApi());

    // Create sampler.
    LiteRtTopKOpenClSampler_Sampler* sampler = nullptr;
    char* error_msg = nullptr;
    int error_code = capi->create_func(env, batch_size, vocab_size,
                                       activation_data_type.has_value()
                                           ? &activation_data_type.value()
                                           : nullptr,
                                       &sampler_params, &sampler, &error_msg);
    RETURN_IF_ERROR(CreateStatusAndFreeErrorMsg(error_code, error_msg));
    ABSL_CHECK(sampler);
    return absl::WrapUnique(
        new TopKOpenClCApiSampler(std::move(capi), sampler));
  }

  ~TopKOpenClCApiSampler() override { capi_->destroy_func(sampler_); }

  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override {
    char* error_msg = nullptr;
    LiteRtTensorBuffer scores_tensor_capi = nullptr;
    if (scores_tensor != nullptr) {
      scores_tensor_capi = scores_tensor->Get();
    }
    int error_code = capi_->sample_func(
        sampler_, logits_tensor.Get(), ids_tensor.Get(),
        scores_tensor_capi ? &scores_tensor_capi : nullptr, &error_msg);
    return CreateStatusAndFreeErrorMsg(error_code, error_msg);
  }

 private:
  using LiteRtTopKOpenClSampler_Sampler = void;
  using LiteRtTopKOpenClSampler_ActivationDataType = void;
  using LiteRtTopKOpenClSampler_SamplerParameters = void;
  using LiteRtTopKOpenClSampler_Create =
      int (*)(LiteRtEnvironment env, int batch_size, int vocab_size,
              const LiteRtTopKOpenClSampler_ActivationDataType* absl_nullable
                  activation_data_type,
              const LiteRtTopKOpenClSampler_SamplerParameters* absl_nullable
                  sampler_params,
              LiteRtTopKOpenClSampler_Sampler** sampler_out,
              char** absl_nullable error_msg);
  using LiteRtTopKOpenClSampler_Destroy =
      void (*)(LiteRtTopKOpenClSampler_Sampler* sampler);
  using LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer =
      int (*)(LiteRtTopKOpenClSampler_Sampler* sampler,
              LiteRtTensorBuffer logits_tensor, LiteRtTensorBuffer ids_tensor,
              const LiteRtTensorBuffer* absl_nullable scores_tensor,
              char** absl_nullable error_msg);

  struct TopKOpenClSamplerCApi {
    SharedLibrary lib;
    LiteRtTopKOpenClSampler_Create create_func;
    LiteRtTopKOpenClSampler_Destroy destroy_func;
    LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer sample_func;

    TopKOpenClSamplerCApi(
        SharedLibrary lib, LiteRtTopKOpenClSampler_Create create_func,
        LiteRtTopKOpenClSampler_Destroy destroy_func,
        LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer sample_func)
        : lib(std::move(lib)),
          create_func(create_func),
          destroy_func(destroy_func),
          sample_func(sample_func) {}
  };

  TopKOpenClCApiSampler(std::unique_ptr<TopKOpenClSamplerCApi> capi,
                        LiteRtTopKOpenClSampler_Sampler* sampler)
      : capi_(std::move(capi)), sampler_(sampler) {}

  static absl::StatusOr<std::unique_ptr<TopKOpenClSamplerCApi>>
  GetTopKOpenClSamplerCApi() {
    // Load Sampler C API library and get the symbols.
    auto maybe_lib = SharedLibrary::Load("libLiteRtTopKOpenClSampler.so",
                                         RtldFlags::Lazy().Local());
    if (!maybe_lib.HasValue()) {
      maybe_lib = SharedLibrary::Load(RtldFlags::kDefault);
    }
    // Note: the Load(kDefault) overload always succeeds, so we are sure that
    // maybe_lib contains a value.
    SharedLibrary lib(std::move(maybe_lib.Value()));
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto sampler_create_func,
        lib.LookupSymbol<LiteRtTopKOpenClSampler_Create>(
            "LiteRtTopKOpenClSampler_Create"));
    RET_CHECK_NE(sampler_create_func, nullptr)
        << "Failed to load LiteRtTopKOpenClSampler_Create";
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto sampler_destroy_func,
        lib.LookupSymbol<LiteRtTopKOpenClSampler_Destroy>(
            "LiteRtTopKOpenClSampler_Destroy"));
    RET_CHECK_NE(sampler_destroy_func, nullptr)
        << "Failed to load LiteRtTopKOpenClSampler_Destroy";
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto sampler_sample_func,
        lib.LookupSymbol<LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer>(
            "LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer"));
    RET_CHECK_NE(sampler_sample_func, nullptr)
        << "Failed to load LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer";
    return std::make_unique<TopKOpenClSamplerCApi>(
        std::move(lib), sampler_create_func, sampler_destroy_func,
        sampler_sample_func);
  }

  std::unique_ptr<TopKOpenClSamplerCApi> capi_;
  LiteRtTopKOpenClSampler_Sampler* const sampler_;
};

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

absl::StatusOr<std::unique_ptr<Sampler>> CreateOpenClSampler(
    int batch_size, proto::SamplerParameters sampler_params,
    LiteRtEnvironment env, int vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  return TopKOpenClCApiSampler::Create(env, batch_size, vocab_size,
                                       activation_data_type, sampler_params);
}
}  // namespace

absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    LiteRtEnvironment env, std::optional<int> vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  switch (backend) {
    case Backend::GPU: {
      RET_CHECK(env != nullptr)
          << "LiteRT environment is needed for GPU sampling.";
      RET_CHECK(vocab_size.has_value())
          << "Vocabulary size is needed for GPU sampling.";
      auto sampler_or =
          CreateOpenClSampler(batch_size, sampler_params, env,
                              vocab_size.value(), activation_data_type);
      if (sampler_or.ok() ||
          sampler_or.status().code() != absl::StatusCode::kUnavailable) {
        // For a normal failure or success, return the result.
        return sampler_or;
      }
    }
      // For a failure due to GPU sampler unavailable, fall back to CPU.
      ABSL_LOG(WARNING)
          << "GPU sampler unavailable. Falling back to CPU sampling. To use "
             "GPU sampling, please make sure libLiteRtTopKOpenClSampler.so is "
             "available at LD_LIBRARY_PATH on device. You can find the shared "
             "library under prebuilt/";
      ABSL_FALLTHROUGH_INTENDED;
    case Backend::CPU:
      return CreateCpuSampler(batch_size, sampler_params);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ", backend));
  }
}
}  // namespace litert::lm
