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

// TODO(b/417209286): Remove this once the model assets are stored in the
// litertlm file format.
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/core/session_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/file_format_util.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {
namespace {

// Builds the LiteRT compiled model executor.
absl::StatusOr<std::unique_ptr<LlmExecutor>> BuildLitertCompiledModelExecutor(
    LlmExecutorSettings executor_settings, ModelResources& model_resources) {
  if (executor_settings.GetModelAssets().HasScopedFile()) {
    return absl::InvalidArgumentError("Model must be passed as a single path.");
  }

  // Create executor that creates and owns the interpreter and kv cache.
  return LlmLiteRtCompiledModelExecutor::Create(std::move(executor_settings),
                                                model_resources);
}

// Assume the files are in the same directory with the following names. This
// should be cleaned up once we store everything in the litertlm file format.
// TODO(b/417209286): Remove this once the model assets are stored in the
// litertlm file format.
constexpr absl::string_view kAuxiliaryModelName =
    "static_a16w4-for-aux_qpa_quantized_gemma3_npu_auxiliary.tflite";
constexpr absl::string_view kEmbedderName =
    "static_a16w4-for-embedder_qpa_quantized_gemma3_npu_embedder.tflite";
constexpr absl::string_view kVocabName = "gemma3_tokenizer.spiece";
using ::odml::infra::LlmLiteRtNpuCompiledModelExecutor::ModelQuantization::
    kAllQuantized;

}  // namespace

class EngineImpl : public Engine {
 public:
  ~EngineImpl() override {
    ABSL_QCHECK_OK(WaitUntilDone(Engine::kDefaultTimeout));
  }

  explicit EngineImpl(EngineSettings engine_settings)
      : engine_settings_(std::move(engine_settings)) {
    if (engine_settings_.IsBenchmarkEnabled()) {
      benchmark_info_ = std::make_optional<BenchmarkInfo>(
          engine_settings_.GetBenchmarkParams().value());
      ABSL_CHECK_OK(
          benchmark_info_->TimeInitPhaseStart("Executor initialization"));
    }
    if ((engine_settings_.GetMainExecutorSettings().GetBackend() ==
         Backend::CPU) ||
        (engine_settings_.GetMainExecutorSettings().GetBackend() ==
         Backend::GPU)) {
      auto& model_assets = engine_settings_.GetMutableMainExecutorSettings()
                               .GetMutableModelAssets();
      auto model_resources = BuildLiteRtCompiledModelResources(model_assets);
      ABSL_CHECK_OK(model_resources);
      litert_model_resources_ = std::move(*model_resources);
      auto scoped_file = model_assets.GetOrCreateScopedFile();
      ABSL_CHECK_OK(scoped_file);

      auto file_format = GetFileFormat(/*model_path=*/"", *scoped_file);
      ABSL_CHECK_OK(file_format);
      // TODO(b/397975034): factor out the tokenizer creation logic once the
      // model loading mechanism of the new file format is determined.
      if (*file_format != FileFormat::TASK &&
          *file_format != FileFormat::LITERT_LM) {
        ABSL_LOG(FATAL) << "Not supported file format: " << *file_format;
      }
      auto tokenizer = litert_model_resources_->GetTokenizer();
      ABSL_CHECK_OK(tokenizer) << tokenizer.status();
      auto llm_metadata = litert_model_resources_->GetLlmMetadata();
      ABSL_CHECK_OK(llm_metadata) << llm_metadata.status();
      // Update and load the parameters from the model file and convert the
      // tokens to ids.
      ABSL_CHECK_OK(
          engine_settings_.MaybeUpdateAndValidate(**tokenizer, *llm_metadata));

      auto executor = BuildLitertCompiledModelExecutor(
          engine_settings_.GetMainExecutorSettings(), *litert_model_resources_);
      ABSL_QCHECK_OK(executor);
      executor_ = std::move(*executor);
      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Executor initialization"));
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseStart("Tokenizer initialization"));
      }

      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Tokenizer initialization"));
      }
    } else {
      std::string model_path(engine_settings_.GetMainExecutorSettings()
                                 .GetModelAssets()
                                 .GetPath()
                                 .value_or(""));

      std::filesystem::path path(model_path);
      ABSL_CHECK(std::filesystem::exists(path));
      auto embedder_path =
          std::filesystem::path(model_path).parent_path() /
          std::string(kEmbedderName);
      ABSL_CHECK(std::filesystem::exists(embedder_path));
      auto auxiliary_path =
          std::filesystem::path(model_path).parent_path() /
          std::string(kAuxiliaryModelName);
      ABSL_CHECK(std::filesystem::exists(auxiliary_path));
      auto vocab_path =
          std::filesystem::path(model_path).parent_path() /
          std::string(kVocabName);

      ABSL_CHECK(std::filesystem::exists(vocab_path));
      auto tokenizer = litert::lm::SentencePieceTokenizer::CreateFromFile(
          vocab_path.string());
      ABSL_CHECK_OK(tokenizer);
      // Update and load the parameters from the model file and convert the
      // tokens to ids.
      ABSL_CHECK_OK(
          engine_settings_.MaybeUpdateAndValidate(**tokenizer, nullptr));
      tokenizer_without_model_resources_ = std::move(*tokenizer);

      auto executor = odml::infra::LlmLiteRtNpuCompiledModelExecutor::Create(
          kAllQuantized, model_path, embedder_path.string(),
          auxiliary_path.string(), path.parent_path().string());
      ABSL_CHECK_OK(executor);
      executor_ = std::move(executor.value());
      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Executor initialization"));
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseStart("Tokenizer initialization"));
      }
      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Tokenizer initialization"));
      }
    }

    // Creating the thread pool of a single thread to execute the works.
    worker_thread_pool_ = std::make_unique<ThreadPool>(/*name_prefix=*/"engine",
                                                       /*max_num_threads=*/1);
  }

  // Method to create the Session.
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) const override {
    SessionConfig config = session_config;
    // TODO(b/418794726): Move this logics to be part of the SessionConfig
    // class.
    RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));  // NOLINT
    auto* tokenizer = tokenizer_without_model_resources_.get();
    if (tokenizer == nullptr) {
      ABSL_CHECK(litert_model_resources_ != nullptr);
      ASSIGN_OR_RETURN(tokenizer, litert_model_resources_->GetTokenizer());  // NOLINT
    }
    return InitializeSession(executor_.get(), tokenizer, config,
                             benchmark_info_, worker_thread_pool_.get());
  }
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return worker_thread_pool_->WaitUntilDone(timeout);
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;
  // Shared executor for all sessions.
  std::unique_ptr<LlmExecutor> executor_;
  // Default stop token ids for all sessions loaded from the model file.
  std::vector<std::vector<int>> stop_token_ids_;
  std::unique_ptr<ModelResources> litert_model_resources_;
  // It's for NPU path. Once NPU read litertlm file, this will be removed.
  std::unique_ptr<SentencePieceTokenizer> tokenizer_without_model_resources_;
  proto::SamplerParameters sampler_params_;

  // Benchmark info for the engine.
  std::optional<BenchmarkInfo> benchmark_info_;

  // Thread pool for the engine to execute the works.
  std::unique_ptr<ThreadPool> worker_thread_pool_;
};

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> Engine::CreateEngine(
    EngineSettings settings_struct) {
  auto llm_impl = std::make_unique<EngineImpl>(std::move(settings_struct));
  return llm_impl;
};

}  // namespace litert::lm
