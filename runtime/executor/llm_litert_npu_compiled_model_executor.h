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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"

namespace odml::infra {

// Component intended to be used with an NPU variant of Gemma3.
class LlmLiteRtNpuCompiledModelExecutor : public ::litert::lm::LlmExecutor {
 public:
  // Holds the latency breakdown stats for the executor.
  // TODO(b/405424188): Use 'litert::lm::BenchmarkInfo' instead.
  struct LatencyStats {
    uint64_t prefill_e2e_latency_us = 0;
    int prefill_num_tokens = 0;
    uint64_t prefill_prepare_input_latency_us = 0;
    uint64_t prefill_embedder_inference_latency_us = 0;
    uint64_t prefill_mask_inference_latency_us = 0;
    uint64_t prefill_rope_inference_latency_us = 0;
    uint64_t prefill_llm_inference_latency_us = 0;
    uint64_t prefill_cache_update_inference_latency_us = 0;

    uint64_t decode_e2e_latency_us = 0;
    int decode_num_tokens = 0;
    uint64_t decode_prepare_input_latency_us = 0;
    uint64_t decode_embedder_inference_latency_us = 0;
    uint64_t decode_mask_inference_latency_us = 0;
    uint64_t decode_rope_inference_latency_us = 0;
    uint64_t decode_llm_inference_latency_us = 0;
    uint64_t decode_cache_update_inference_latency_us = 0;
    uint64_t decode_sampling_latency_us = 0;
  };

  // Creates an executor from the resources.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  Create(
      const litert::lm::LlmExecutorSettings& executor_settings,
      litert::lm::ModelResources& resources,
      const std::optional<std::string>& dispatch_library_path = std::nullopt);

  // Input APIs:
  // Basic API to trigger the "prefill" or "prefix" process.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(const ::litert::lm::ExecutorInputs& inputs) override;

  // Advanced API to allow customized query parameters.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(
      const ::litert::lm::ExecutorInputs& inputs,
      const ::litert::lm::ExecutorPrefillParams& params) override;

  // Output APIs:
  // Basic API to trigger the "decode" process.
  absl::Status Decode(::litert::TensorBuffer& output_tokens) override;

  // Basic API to trigger the "decode" process but without sampling.
  // Input is token ids with shape `[batch, sequence_length]`
  // Output is logits with shape `[batch, sequence_length, vocab_size]`
  // TODO: b/355310550 - Shall we change the function naming here to not
  // overload Decode?
  absl::Status Decode(const ::litert::lm::ExecutorInputs& inputs,
                      ::litert::TensorBuffer& output_logits) override;

  absl::string_view ExecutorBackendName() const override {
    return "LiteRT NPU Compiled Model";
  }

  // Gets the current step of the executor.
  // Public API, the return value is the current step that user expects (e.g.
  // users prefill 100 tokens, then they expect the current step to be 100). It
  // is different from the internal current step.
  absl::StatusOr<int> GetCurrentStep() const override {
    return current_step_ + (next_input_token_id_ == -1 ? 0 : 1);
  }

  absl::StatusOr<int> GetVocabSize() override;

  absl::StatusOr<litert::lm::LlmExecutorSettings> GetExecutorSettings()
      const override {
    return executor_settings_;
  };
  // Prints the latency stats for the executor.  Intended to be used for
  // profiling.
  LatencyStats GetLatencyStats() const;

 private:
  // Holds the tensor buffers maps for the inference of a precompiled model,
  // both for prefill and decode.
  struct InferenceContext {
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        prefill_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        prefill_output_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_output_buffers;
    InferenceContext(
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_output_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_output_buffers);
  };

  // Holds the context for the embedder model.
  struct EmbedderContext {
    ::litert::Model embedder_model;
    ::litert::CompiledModel embedder_compiled_model;
    InferenceContext inference_context;
    EmbedderContext(
        ::litert::CompiledModel embedder_compiled_model,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_output_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_output_buffers);
  };

  // Holds the context for the NPU auxiliary model, which contains several
  // signatures for Mask, RoPE and KV cache update computation.
  struct NpuAuxiliaryContext {
    ::litert::CompiledModel npu_auxiliary_compiled_model;
    NpuAuxiliaryContext(::litert::CompiledModel npu_auxiliary_compiled_model);
  };

 protected:
  LlmLiteRtNpuCompiledModelExecutor(
      litert::lm::LlmExecutorSettings executor_settings,
      EmbedderContext embedder_context,
      NpuAuxiliaryContext npu_auxiliary_context, InferenceContext mask_context,
      InferenceContext rope_context, ::litert::Environment llm_env,
      ::litert::CompiledModel llm_compiled_model,
      InferenceContext llm_inference_context,
      InferenceContext cache_update_inference_context,
      ::litert::lm::SortedPrefillSignatureMap prefill_signature_map)
      : executor_settings_(std::move(executor_settings)),
        embedder_context_(std::move(embedder_context)),
        npu_auxiliary_context_(std::move(npu_auxiliary_context)),
        mask_context_(std::move(mask_context)),
        rope_context_(std::move(rope_context)),
        env_(std::move(llm_env)),
        llm_compiled_model_(std::move(llm_compiled_model)),
        llm_inference_context_(std::move(llm_inference_context)),
        cache_update_inference_context_(
            std::move(cache_update_inference_context)),
        prefill_signature_map_(std::move(prefill_signature_map)) {
    executor_settings_.SetMaxNumTokens(1280);
  }

 private:
  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length.
  absl::Status PrefillInternal(absl::string_view prefill_signature,
                               absl::Span<const int> ids);

  // Decode internal implementation, without result downloading.
  // Caller of this function is responsible for capturing the output.
  absl::Status DecodeInternal(::litert::lm::ExecutorInputs inputs);

  // Creates the context for the embedder model.  Instead of creating new
  // output buffers for the embedder, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers' and
  // 'gemma_decode_input_buffers'.  Similarly, instead of creating the buffers
  // for the input tokens the provided 'prefill_input_tokens' and
  // 'decode_input_tokens' will be used.
  static absl::StatusOr<EmbedderContext> CreateEmbedderContextWithBufferSharing(
      ::litert::Environment& env, const litert::Model& embedder_model,
      ::litert::TensorBuffer prefill_input_tokens,
      ::litert::TensorBuffer decode_input_tokens,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers);

  // Creates the context for the NPU auxiliary model.
  static absl::StatusOr<NpuAuxiliaryContext> CreateNpuAuxiliaryContext(
      ::litert::Environment& env, const litert::Model& npu_auxiliary_model);

  // Creates the context for the mask model.  Instead of creating new
  // output buffers for the mask model, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers' and
  // 'gemma_decode_input_buffers'.
  static absl::StatusOr<InferenceContext> CreateMaskContextWithBufferSharing(
      NpuAuxiliaryContext& npu_auxiliary_context,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers);

  // Creates the context for the RoPE model.  Instead of creating new
  // output buffers for the RoPE model, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers' and
  // 'gemma_decode_input_buffers'.
  static absl::StatusOr<InferenceContext> CreateRopeContextWithBufferSharing(
      NpuAuxiliaryContext& npu_auxiliary_context,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers);

  // Creates the context for the LLM model.
  static absl::StatusOr<InferenceContext>
  CreateLlmInferenceContextWithBufferSharing(
      ::litert::Environment& env, ::litert::CompiledModel& llm_compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers);

  static absl::StatusOr<InferenceContext>
  CreateCacheUpdateInferenceContextWithBufferSharing(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      ::litert::TensorBuffer prefill_input_pos,
      ::litert::TensorBuffer decode_input_pos);
  // Run a 'warmup' inference on every model (prefill and decode).  This is
  // intended to be called before the first actual inference.
  static absl::Status WarmupInference(
      ::litert::CompiledModel& compiled_model_llm,
      const InferenceContext& llm_inference_context,
      ::litert::CompiledModel& compiled_model_auxiliary,
      const InferenceContext& rope_inference_context,
      const InferenceContext& mask_inference_context,
      const InferenceContext& cache_update_inference_context);

  litert::lm::LlmExecutorSettings executor_settings_;
  LatencyStats latency_stats_;
  EmbedderContext embedder_context_;
  NpuAuxiliaryContext npu_auxiliary_context_;
  InferenceContext mask_context_;
  InferenceContext rope_context_;
  ::litert::Environment env_;
  ::litert::CompiledModel llm_compiled_model_;

  InferenceContext llm_inference_context_;
  InferenceContext cache_update_inference_context_;
  ::litert::lm::SortedPrefillSignatureMap prefill_signature_map_;

  // The sampled ids to use for external sampling.
  // The layout is batch-major.
  // e.g. for output_batch_size=2, the layout is:
  // {batch_0_seq_0, batch_1_seq_0, batch_0_seq_1, batch_1_seq_1, ...}
  std::vector<int> sampled_ids_;
  // Output batch size for the sampled ids.
  int output_batch_size_ = 0;

  // Internal timestep.
  int current_step_ = 0;

  // TODO: b/404625243 - To be implemented.
  // The processed tokens.
  std::vector<int> processed_tokens_;

  // The token served as the first input token to the model for next Prefill or
  // Decode.
  int next_input_token_id_ = -1;
};

}  // namespace odml::infra

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_
