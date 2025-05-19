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
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/framework/thread_options.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {
namespace {

// Builds the LiteRT compiled model executor.
absl::StatusOr<std::unique_ptr<LlmExecutor>> BuildLitertCompiledModelExecutor(
    const std::unique_ptr<ModelResources>& model_resources,
    const LlmExecutorSettings& executor_settings) {
  if (executor_settings.GetModelAssets().HasScopedFile()) {
    return absl::InvalidArgumentError("Model must be passed as a single path.");
  }

  // Create executor that creates and owns the interpreter and kv cache.
  std::unique_ptr<LlmExecutor> executor;
  ASSIGN_OR_RETURN(executor,  // NOLINT
                   LlmLiteRtCompiledModelExecutor::Create(
                       executor_settings, std::move(model_resources)));
  return executor;
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

  explicit EngineImpl(const EngineSettings& engine_settings) {
    if (engine_settings.IsBenchmarkEnabled()) {
      benchmark_info_ = std::make_optional<BenchmarkInfo>(
          engine_settings.GetBenchmarkParams().value());
      ABSL_CHECK_OK(
          benchmark_info_->TimeInitPhaseStart("Executor initialization"));
    }
    auto model_path_view =
        engine_settings.GetMainExecutorSettings().GetModelAssets().GetPath();
    ABSL_CHECK_OK(model_path_view);
    std::string model_path(*model_path_view);
    if ((engine_settings.GetMainExecutorSettings().GetBackend() ==
         Backend::CPU) ||
        (engine_settings.GetMainExecutorSettings().GetBackend() ==
         Backend::GPU)) {
      auto model_resources = BuildLiteRtCompiledModelResources(model_path);
      ABSL_CHECK_OK(model_resources);
      litert_model_resources_ = std::move(*model_resources);
      auto executor = BuildLitertCompiledModelExecutor(
          litert_model_resources_, engine_settings.GetMainExecutorSettings());

      ABSL_QCHECK_OK(executor);
      executor_ = std::move(*executor);
      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Executor initialization"));
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseStart("Tokenizer initialization"));
      }
      // TODO(b/397975034): factor out the tokenizer creation logic once the
      // model loading mechanism of the new file format is determined.
      auto scoped_file = ScopedFile::Open(model_path);
      ABSL_CHECK_OK(scoped_file);

      // TODO(b/413793273): Read the header bytes to determine the file type
      // instead of depending on the file extension.
      if (absl::EndsWith(model_path, ".litertlm")) {
        ABSL_LOG(FATAL) << "Not supported file format in OSS yet.";
      } else {
        auto resources = ModelAssetBundleResources::Create(
            /*tag=*/"", *std::move(scoped_file));
        auto vocab_buffer = (*resources)->GetFile("TOKENIZER_MODEL");
        tokenizer_ =
            std::move(*SentencePieceTokenizer::CreateFromBuffer(*vocab_buffer));
      }
      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Tokenizer initialization"));
      }
    } else {
      std::filesystem::path path(model_path);
      ABSL_CHECK(std::filesystem::exists(path));
      const std::string embedder_path =
          std::filesystem::path(model_path).parent_path() /
          std::string(kEmbedderName);
      ABSL_CHECK(std::filesystem::exists(embedder_path));
      const std::string auxiliary_path =
          std::filesystem::path(model_path).parent_path() /
          std::string(kAuxiliaryModelName);
      ABSL_CHECK(std::filesystem::exists(auxiliary_path));
      const std::string vocab_path =
          std::filesystem::path(model_path).parent_path() /
          std::string(kVocabName);
      ABSL_CHECK(std::filesystem::exists(vocab_path));
      auto executor_or = odml::infra::LlmLiteRtNpuCompiledModelExecutor::Create(
          kAllQuantized, model_path, embedder_path, auxiliary_path,
          std::string(path.parent_path()));
      ABSL_CHECK_OK(executor_or);
      executor_ = std::move(executor_or.value());
      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Executor initialization"));
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseStart("Tokenizer initialization"));
      }
      auto tokenizer_or =
          litert::lm::SentencePieceTokenizer::CreateFromFile(vocab_path);
      ABSL_CHECK_OK(tokenizer_or);
      tokenizer_ = std::move(tokenizer_or.value());
      if (benchmark_info_.has_value()) {
        ABSL_CHECK_OK(
            benchmark_info_->TimeInitPhaseEnd("Tokenizer initialization"));
      }
    }

    // TODO(b/397975034) Add support for stop tokens loading from the model
    // file, most likely by creating a simplified
    // DeriveLlmModelSettingsStruct.
    AddStopTokenIds("<eos>");
    AddStopTokenIds("<end_of_turn>");
    // TODO(b/412390852): Add logics to initialize the sampler.

    // Creating the thread pool of a single thread to execute the works.
    auto thread_pool = ThreadPool::CreateThreadPool(ThreadOptions(),
                                                    /*name_prefix=*/"engine",
                                                    /*num_threads=*/1);
    ABSL_CHECK_OK(thread_pool);
    worker_thread_pool_ = std::move(*thread_pool);
    worker_thread_pool_->StartWorkers();
  }

  // Method to create the Session.
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) const override {
    return InitializeSession(executor_, tokenizer_, stop_token_ids_,
                             session_config, benchmark_info_,
                             worker_thread_pool_);
  }
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return worker_thread_pool_->WaitUntilDone(timeout);
  }

 private:
  void AddStopTokenIds(const std::string& stop_token) {
    auto stop_token_ids = tokenizer_->TextToTokenIds(stop_token);
    if ((*stop_token_ids).size() == 1) {
      stop_token_ids_.push_back((*stop_token_ids)[0]);
    } else {
      ABSL_LOG(ERROR) << "Stop token \"" << stop_token
                      << "\" maps to multiple token ids: "
                      << (*stop_token_ids).size();
    }
  }

  // Shared executor for all sessions.
  std::shared_ptr<LlmExecutor> executor_;
  // Shared tokenizer for all sessions.
  std::shared_ptr<Tokenizer> tokenizer_;
  // Default stop token ids for all sessions loaded from the model file.
  std::vector<int> stop_token_ids_;
  std::unique_ptr<ModelResources> litert_model_resources_;
  proto::SamplerParameters sampler_params_;

  // Benchmark info for the engine.
  std::optional<BenchmarkInfo> benchmark_info_;

  // Thread pool for the engine to execute the works.
  std::shared_ptr<ThreadPool> worker_thread_pool_;
};

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> Engine::CreateEngine(
    const EngineSettings& settings_struct) {
  auto llm_impl = std::make_unique<EngineImpl>(settings_struct);
  return llm_impl;
};

}  // namespace litert::lm
