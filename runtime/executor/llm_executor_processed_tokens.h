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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_PROCESSED_TOKENS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_PROCESSED_TOKENS_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Information which is used to process a token.
class TokenData {
 public:
  explicit TokenData(int token_id) : id_(token_id) {}

  TokenData(int token_id, std::vector<float> token_embedding,
            std::vector<float> token_per_layer_embedding)
      : id_(token_id),
        embedding_(std::move(token_embedding)),
        per_layer_embedding_(std::move(token_per_layer_embedding)) {}

  int id() const { return id_; }

  absl::Span<const float> embedding() const { return embedding_; }
  std::vector<float>& mutable_embedding() { return embedding_; }

  absl::Span<const float> per_layer_embedding() const {
    return per_layer_embedding_;
  }
  std::vector<float>& mutable_per_layer_embedding() {
    return per_layer_embedding_;
  }

 private:
  // The token id that is to be processed.
  const int id_;

  // May contain the embedding corresponding to the token id.
  std::vector<float> embedding_;

  // May contain the per-layer embedding corresponding to the token id.
  std::vector<float> per_layer_embedding_;
};

// Keeps track of processed tokens during the LLM execution.
//
// This class is used by `ProcessedContext` to store the sequence of tokens
// that have been processed so far. It keeps track of both the processed tokens
// and a pending input token, if any, which may be used by backends which
// require an input token to be provided during Decode.
class ProcessedTokens {
 public:
  // A token and its corresponding step. The token is std::nullopt if the step
  // does not correspond to a token in this ProcessedTokens.
  struct StepAndToken {
    int step;
    std::shared_ptr<TokenData> token = nullptr;
  };

  ProcessedTokens() = default;

  ProcessedTokens(const ProcessedTokens&) = default;
  ProcessedTokens(ProcessedTokens&&) noexcept = default;
  ProcessedTokens& operator=(const ProcessedTokens&) = default;
  ProcessedTokens& operator=(ProcessedTokens&&) noexcept = default;

  // Returns the number of processed tokens inclusive of the pending input
  // token, if any.
  int TokenCount() const {
    return tokens_.size() + (pending_input_token_ ? 1 : 0);
  }

  // Returns `pending_input_token_` and its step, if it exists; otherwise,
  // the step after the last processed token.
  StepAndToken GetNextUnprocessedToken() const {
    // This cast should not overflow since INT_MAX is more than 2 billion, which
    // far exceeds the maximum context length for current on-device LLMs.
    int step = static_cast<int>(tokens_.size());
    return StepAndToken{.step = step, .token = pending_input_token_};
  }

  // Appends the given tokens to the list of processed tokens.
  void AddProcessedTokens(std::vector<int> token_ids) {
    tokens_.insert(tokens_.end(), token_ids.begin(), token_ids.end());
  }

  // Add a token as a "pending" input token, which indicates that the token has
  // not yet been processed by the LLM, but is part of the current context and
  // is to be processed during the next Prefill or Decode step. This may be used
  // by backends which require an input token to be provided during Decode.
  absl::Status AddPendingInputToken(std::shared_ptr<TokenData> token) {
    if (pending_input_token_ != nullptr) {
      return absl::InternalError(
          "AddPendingInputToken called with an existing pending token.");
    }
    pending_input_token_ = std::move(token);
    return absl::OkStatus();
  }

  // Reverts the processed tokens to the given step. This new step must be
  // non-negative and smaller than the current token count.
  absl::Status RollBackToStep(int new_step) {
    if (new_step < 0) {
      return absl::InternalError(
          absl::StrCat("new_step must be non-negative, got ", new_step));
    }
    if (new_step > TokenCount()) {
      return absl::InternalError(absl::StrCat(
          "new_step must be less than or equal to TokenCount(), got ", new_step,
          " vs ", TokenCount()));
    }

    if (new_step == TokenCount()) {
      return absl::OkStatus();
    }

    pending_input_token_ = nullptr;
    tokens_.resize(new_step);
    return absl::OkStatus();
  }

  // Returns the token at the given `step` or std::nullopt if the step does not
  // correspond to a token.
  std::optional<int> GetTokenAtStep(int step) const {
    if (step < 0 || step >= TokenCount()) {
      return std::nullopt;
    }
    if (step == tokens_.size() && pending_input_token_) {
      return pending_input_token_->id();
    }

    return tokens_[step];
  }

  // Marks the pending input token as processed. It is an error to call this
  // function if there is no pending input token.
  absl::Status MarkPendingInputTokenAsProcessed() {
    if (!pending_input_token_) {
      return absl::InternalError(
          "MarkPendingInputTokenAsProcessed called with no pending token.");
    }
    tokens_.push_back(pending_input_token_->id());
    pending_input_token_ = nullptr;
    return absl::OkStatus();
  }

  // Returns a deep copy of the complete list of processed tokens, inclusive of
  // the pending input token, if any.
  std::vector<int> GetCopyOfTokens() const {
    std::vector<int> copy_of_tokens = tokens_;
    if (pending_input_token_) {
      copy_of_tokens.push_back(pending_input_token_->id());
    }
    return copy_of_tokens;
  }

  // WARNING: This function returns a reference to the internal `tokens_`
  // directly, which may not include the pending input token. This method MUST
  // NOT be used in code that runs a backend which uses a pending input token.
  const std::vector<int>& GetTokensUnsafe() const {
    ABSL_CHECK_EQ(pending_input_token_, nullptr);
    return tokens_;
  }

 private:
  std::vector<int> tokens_;

  std::shared_ptr<TokenData> pending_input_token_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_PROCESSED_TOKENS_H_
