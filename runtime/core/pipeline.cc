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

#include "runtime/core/pipeline.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {
namespace {

// TODO(b/423364170): all LLM Executors should respect the max number of tokens
// returned by the model. We should remove this default value once all Executors
// are compliant with the max number of tokens.
constexpr int kDefaultMaxNumTokens = 4096;
int TryGetMaxNumTokens(std::shared_ptr<LlmExecutor> executor) {
  auto settings = executor->GetExecutorSettings();
  if (!settings.ok()) {
    // If the executor settings are not available, we will use the default
    // value.
    ABSL_LOG(WARNING) << "Failed to get executor settings: "
                      << settings.status();
    return kDefaultMaxNumTokens;
  }
  return settings->GetMaxNumTokens();
}

// Check whether the decoding loop should stop.
bool ShouldStop(bool hit_stop_tokens, int benchmark_decode_token_count,
                int num_decoded_steps, int current_step, int max_num_tokens,
                InferenceObservable* observer) {
  // Stopping conditions.
  if (hit_stop_tokens && benchmark_decode_token_count == 0) {
    // Only early stop if no decode step
    // is requested by benchmark.
    return true;
  } else if (benchmark_decode_token_count > 0 &&
             num_decoded_steps >= benchmark_decode_token_count) {
    // Stop when the number of decode steps is equal to the
    // benchmark_decode_token_count (when specified).
    return true;
  } else if (current_step >= max_num_tokens) {
    // Reaching maximum number of kv-cache size.
    if (observer != nullptr) {
      observer->OnError(absl::InternalError("Maximum kv-cache size reached."));
    }
    return true;
  }
  return false;
}

// A wrapper class to run one step of the decode process. It allows us to reduce
// the code duplication between different decode functions.
// TODO(b/417568021): Refactor the class to make it more readable.
class DecodeExternalSampling {
 public:
  DecodeExternalSampling(std::shared_ptr<LlmExecutor> executor,
                         std::shared_ptr<Tokenizer> tokenizer,
                         int num_output_candidates, Sampler& sampler,
                         const StopTokenDetector& stop_token_detector,
                         std::optional<BenchmarkInfo>& benchmark_info)
      : executor_(executor),
        tokenizer_(tokenizer),
        num_output_candidates_(num_output_candidates),
        sampler_(sampler),
        benchmark_info_(benchmark_info),
        stop_token_detector_(stop_token_detector) {
    auto scores_tensor = CreateTensorBuffer<float>({num_output_candidates_});
    scores_tensor_ = std::move(*scores_tensor);
  }

  // Runs one step of the decode process with sampling done externally from the
  // Executor.
  absl::StatusOr<bool> Run(litert::TensorBuffer& decoded_ids) {
    LITERT_ASSIGN_OR_RETURN(auto duplicate_decoded_ids,
                            decoded_ids.Duplicate());
    ExecutorInputs inputs(ExecutorTextData(std::move(duplicate_decoded_ids)),
                          std::nullopt, std::nullopt);
    if (benchmark_info_.has_value()) {
      RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
    }
    ASSIGN_OR_RETURN(auto output_logits, executor_->DecodeLogits(inputs));
    if (benchmark_info_.has_value()) {
      RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
    }
    RETURN_IF_ERROR(sampler_.SampleToIdAndScoreBuffer(
        output_logits, decoded_ids, &scores_tensor_));
    if (benchmark_info_.has_value()) {
      RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
    }
    ASSIGN_OR_RETURN(result_tokens_,
                     tokenizer_->TensorBufferToText(decoded_ids));

    // Update the stop_tokens_found vector with the latest decoded ids.
    LITERT_ASSIGN_OR_RETURN_ABSL(auto decoded_ids_span,
                                 ReferTensorBufferAsSpan<int>(decoded_ids));
    LITERT_ASSIGN_OR_RETURN_ABSL(
        scores_span_, ReferTensorBufferAsSpan<float>(scores_tensor_));
    RETURN_IF_ERROR(stop_token_detector_.ProcessTokens(decoded_ids_span));
    ASSIGN_OR_RETURN(bool hit_stop_tokens, stop_token_detector_.AllDone());
    return hit_stop_tokens;
  }

  absl::Span<float> GetScores() { return scores_span_; }

  const std::vector<std::string>& GetResultTokens() const {
    return result_tokens_;
  }

  const std::vector<bool>& GetStopTokensFound() const {
    return stop_token_detector_.GetStopTokensFound();
  }

 private:
  std::shared_ptr<LlmExecutor> executor_;
  std::shared_ptr<Tokenizer> tokenizer_;
  const int num_output_candidates_;
  Sampler& sampler_;
  std::optional<BenchmarkInfo> benchmark_info_;
  litert::TensorBuffer scores_tensor_;
  std::vector<std::string> result_tokens_;
  absl::Span<float> scores_span_;
  StopTokenDetector stop_token_detector_;
};

// A wrapper class to run one step of the decode process with sampling done
// internally from the Executor.
class DecodeInternalSamplingOneStep {
 public:
  DecodeInternalSamplingOneStep(std::shared_ptr<LlmExecutor> executor,
                                std::shared_ptr<Tokenizer> tokenizer,
                                int num_output_candidates,
                                const StopTokenDetector& stop_token_detector,
                                std::optional<BenchmarkInfo>& benchmark_info)
      : executor_(executor),
        tokenizer_(tokenizer),
        num_output_candidates_(num_output_candidates),
        benchmark_info_(benchmark_info),
        stop_token_detector_(stop_token_detector) {
    stop_tokens_found_ = std::vector<bool>(num_output_candidates_, false);
    auto output_tokens = CreateTensorBuffer<int>({num_output_candidates_, 1});
    output_tokens_ = std::move(*output_tokens);
  }

  // Runs one step of the decode process with sampling done externally from the
  // Executor.
  absl::StatusOr<bool> Run() {
    if (benchmark_info_.has_value()) {
      RETURN_IF_ERROR(
          benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
    }
    RETURN_IF_ERROR(executor_->Decode(output_tokens_));
    if (benchmark_info_.has_value()) {
      RETURN_IF_ERROR(
          benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
    }
    LITERT_ASSIGN_OR_RETURN_ABSL(auto output_tokens_span,
                                 ReferTensorBufferAsSpan<int>(output_tokens_));
    if (output_tokens_span.size() != 1) {
      return absl::InternalError("Unexpected number of decoded tokens.");
    }

    ASSIGN_OR_RETURN(result_tokens_,
                     tokenizer_->TensorBufferToText(output_tokens_));
    RETURN_IF_ERROR(stop_token_detector_.ProcessTokens(output_tokens_span));
    ASSIGN_OR_RETURN(bool hit_stop_tokens, stop_token_detector_.AllDone());
    return hit_stop_tokens;
  }

  absl::Span<float> GetScores() { return scores_span_; }

  const std::vector<std::string>& GetResultTokens() const {
    return result_tokens_;
  }

  const std::vector<bool>& GetStopTokensFound() const {
    return stop_token_detector_.GetStopTokensFound();
  }

 private:
  std::shared_ptr<LlmExecutor> executor_;
  std::shared_ptr<Tokenizer> tokenizer_;
  const int num_output_candidates_;
  std::optional<BenchmarkInfo> benchmark_info_;
  std::vector<bool> stop_tokens_found_;
  litert::TensorBuffer output_tokens_;
  std::vector<std::string> result_tokens_;
  absl::Span<float> scores_span_;
  StopTokenDetector stop_token_detector_;
};

}  // namespace

absl::StatusOr<int> Prefill(std::shared_ptr<LlmExecutor> executor,
                            std::shared_ptr<Tokenizer> tokenizer,
                            absl::string_view prompt, int bos_token_id,
                            bool wait_for_completion,
                            std::optional<BenchmarkInfo>& benchmark_info) {
  int benchmark_prefill_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_prefill_token_count =
        benchmark_info->GetBenchmarkParams().num_prefill_tokens();
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnStart());
  }
  ASSIGN_OR_RETURN(std::vector<int> ids, tokenizer->TextToTokenIds(prompt));
  if (benchmark_prefill_token_count > 0) {
    // If benchmark is enabled, we will use the benchmark prefill token count
    // to set the prefill token count.
    ids.resize(benchmark_prefill_token_count);
  } else {
    ids.insert(ids.begin(), bos_token_id);
  }
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  if (ids.size() >= max_num_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. Exceeding the maximum number of tokens "
        "allowed: ",
        ids.size(), " >= ", max_num_tokens));
  }
  ASSIGN_OR_RETURN(auto ids_buffer, tokenizer->TokenIdsToTensorBuffer(ids));
  LITERT_ASSIGN_OR_RETURN_ABSL(auto ids_buffer_span,
                               ReferTensorBufferAsSpan<int>(ids_buffer));
  if (ids_buffer_span.empty()) {
    return absl::InternalError("Input token ids are empty.");
  }
  const int last_token_id = ids_buffer_span.back();
  ExecutorPrefillParams params;
  params.SetWaitForCompletion(wait_for_completion);
  RETURN_IF_ERROR(
      executor->Prefill(ExecutorInputs(ExecutorTextData(std::move(ids_buffer)),
                                       std::nullopt, std::nullopt),
                        params));
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnEnd(ids_buffer_span.size()));
  }
  return last_token_id;
}

absl::StatusOr<Responses> Decode(std::shared_ptr<LlmExecutor> executor,
                                 std::shared_ptr<Tokenizer> tokenizer,
                                 const StopTokenDetector& stop_token_detector,
                                 std::optional<BenchmarkInfo>& benchmark_info) {
  int benchmark_decode_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_decode_token_count =
        benchmark_info->GetBenchmarkParams().num_decode_tokens();
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
  }
  // Hard code the number of output candidates to 1 for now.
  const int num_output_candidates = 1;
  Responses responses(num_output_candidates);
  std::vector<std::string>& response_texts =
      responses.GetMutableResponseTexts();
  // TODO(b/397975034) LLM Executor should return error when reaching the
  // maximum number of kv-cache steps.
  int num_decoded_steps = 0;
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  DecodeInternalSamplingOneStep run_one_step(
      executor, tokenizer, num_output_candidates, stop_token_detector,
      benchmark_info);
  while (true) {
    auto hit_stop_tokens = run_one_step.Run();
    if (!hit_stop_tokens.ok()) {
      return hit_stop_tokens.status();
    }
    response_texts[0] +=
        absl::StrReplaceAll(run_one_step.GetResultTokens()[0], {{"▁", " "}});
    num_decoded_steps++;

    if (ShouldStop(*hit_stop_tokens, benchmark_decode_token_count,
      num_decoded_steps, executor->GetCurrentStep().value(), max_num_tokens,
      /*observer=*/nullptr)) {
      break;
    }
  }
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(num_decoded_steps *
                                                      num_output_candidates));
  }
  return responses;
}

absl::Status DecodeStreaming(std::shared_ptr<LlmExecutor> executor,
                             std::shared_ptr<Tokenizer> tokenizer,
                             const StopTokenDetector& stop_token_detector,
                             std::optional<BenchmarkInfo>& benchmark_info,
                             InferenceObservable* observer) {
  if (observer == nullptr) {
    return absl::InvalidArgumentError(
        "Observer must be provided for streaming.");
  }
  int benchmark_decode_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_decode_token_count =
        benchmark_info->GetBenchmarkParams().num_decode_tokens();
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
  }
  // Hard code the number of output candidates to 1 for now.
  const int num_output_candidates = 1;
  // TODO(b/397975034) LLM Executor should return error when reaching the
  // maximum number of kv-cache steps.
  int num_decoded_steps = 0;
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  DecodeInternalSamplingOneStep run_one_step(
      executor, tokenizer, num_output_candidates, stop_token_detector,
      benchmark_info);
  while (true) {
    Responses responses(num_output_candidates);
    std::vector<std::string>& response_texts =
        responses.GetMutableResponseTexts();
    auto hit_stop_tokens = run_one_step.Run();
    if (!hit_stop_tokens.ok()) {
      observer->OnError(hit_stop_tokens.status());
      return hit_stop_tokens.status();
    }
    response_texts[0] +=
        absl::StrReplaceAll(run_one_step.GetResultTokens()[0], {{"▁", " "}});
    num_decoded_steps++;
    observer->OnNext(responses);

    if (ShouldStop(*hit_stop_tokens, benchmark_decode_token_count,
      num_decoded_steps, executor->GetCurrentStep().value(), max_num_tokens,
      observer)) {
      break;
    }
  }
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(num_decoded_steps *
                                                      num_output_candidates));
  }
  observer->OnDone();
  return absl::OkStatus();
}

absl::StatusOr<Responses> DecodeCustomSampling(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info) {
  int benchmark_decode_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_decode_token_count =
        benchmark_info->GetBenchmarkParams().num_decode_tokens();
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
  }
  Responses responses(num_output_candidates);
  // TODO(b/397975034) LLM Executor should return error when reaching the
  // maximum number of kv-cache steps.
  std::vector<float>& scores = responses.GetMutableScores();
  std::fill(scores.begin(), scores.end(), 0.0f);
  std::vector<int> num_decoded_tokens(num_output_candidates, 0);
  int num_decode_steps = 0;
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  DecodeExternalSampling run_one_step(executor, tokenizer,
                                      num_output_candidates, sampler,
                                      stop_token_detector, benchmark_info);

  while (true) {
    ASSIGN_OR_RETURN(bool hit_stop_tokens, run_one_step.Run(decoded_ids));

    // Append the results to the final results vector. Note that only the
    // candidates that have not reached the stop token are added to the final
    // results vector.
    std::vector<std::string>& response_texts =
        responses.GetMutableResponseTexts();
    for (int j = 0; j < num_output_candidates; ++j) {
      // Only add the result if the stop token has not been found yet.
      if (!run_one_step.GetStopTokensFound()[j]) {
        // The tokenizer may return a token with a special character "▁" that
        // should be replaced with a space.
        response_texts[j] += absl::StrReplaceAll(
            run_one_step.GetResultTokens()[j], {{"▁", " "}});
        num_decoded_tokens[j]++;
        scores[j] += run_one_step.GetScores()[j];
      }
    }
    num_decode_steps++;
    if (ShouldStop(hit_stop_tokens, benchmark_decode_token_count,
      num_decode_steps, executor->GetCurrentStep().value(), max_num_tokens,
      /*observer=*/nullptr)) {
      break;
    }
  }
  for (int j = 0; j < num_output_candidates; ++j) {
    if (num_decoded_tokens[j] > 0) {
      scores[j] /= num_decoded_tokens[j];
    } else {
      scores[j] = -std::numeric_limits<float>::infinity();
    }
  }
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(num_decode_steps *
                                                      num_output_candidates));
  }
  return responses;
}

absl::Status DecodeCustomSamplingStreaming(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info,
    InferenceObservable* observer) {
  if (observer == nullptr) {
    return absl::InvalidArgumentError(
        "Observer must be provided for streaming.");
  }
  int benchmark_decode_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_decode_token_count =
        benchmark_info->GetBenchmarkParams().num_decode_tokens();
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
  }
  // TODO(b/397975034) LLM Executor should return error when reaching the
  // maximum number of kv-cache steps.
  int num_decode_steps = 0;
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  DecodeExternalSampling run_one_step(executor, tokenizer,
                                      num_output_candidates, sampler,
                                      stop_token_detector, benchmark_info);

  // Enter the loop to run the decode process.
  while (true) {
    absl::StatusOr<bool> hit_stop_tokens = run_one_step.Run(decoded_ids);
    if (!hit_stop_tokens.ok()) {
      observer->OnError(hit_stop_tokens.status());
      return hit_stop_tokens.status();
    }

    Responses responses(num_output_candidates);
    std::vector<float>& scores = responses.GetMutableScores();
    // Append the results to the final results vector. Note that only the
    // candidates that have not reached the stop token are added to the final
    // results vector.
    std::vector<std::string>& response_texts =
        responses.GetMutableResponseTexts();
    for (int j = 0; j < num_output_candidates; ++j) {
      // Only add the result if the stop token has not been found yet.
      if (!run_one_step.GetStopTokensFound()[j]) {
        // The tokenizer may return a token with a special character "▁" that
        // should be replaced with a space.
        response_texts[j] += absl::StrReplaceAll(
            run_one_step.GetResultTokens()[j], {{"▁", " "}});
        scores[j] += run_one_step.GetScores()[j];
      }
    }
    num_decode_steps++;
    observer->OnNext(responses);
    if (ShouldStop(*hit_stop_tokens, benchmark_decode_token_count,
      num_decode_steps, executor->GetCurrentStep().value(), max_num_tokens,
      observer)) {
      break;
    }
  }
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(num_decode_steps *
                                                      num_output_candidates));
  }
  observer->OnDone();
  return absl::OkStatus();
}

}  // namespace litert::lm
