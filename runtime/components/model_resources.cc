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

#include "runtime/components/model_resources.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

absl::StatusOr<std::shared_ptr<litert::Model>>
ModelResources::GetTFLiteModel() {
  if (litert_model_ != nullptr) {
    return litert_model_;
  }
  // Lazily create the model from the task bundle or litertlm file.
  if (litert_lm_loader_ != nullptr) {
    ABSL_LOG(FATAL) << "Not supported file format in OSS yet.";
  } else if (model_asset_bundle_resources_ != nullptr) {
    auto buffer =
        model_asset_bundle_resources_->GetFile("TF_LITE_PREFILL_DECODE");
    ABSL_LOG(INFO) << "litert model size: " << buffer->size();
    auto buffer_ref = BufferRef<uint8_t>(buffer->data(), buffer->size());
    LITERT_ASSIGN_OR_RETURN(auto model, Model::CreateFromBuffer(buffer_ref));
    litert_model_ = std::make_shared<Model>(std::move(model));
    return litert_model_;
  }

  return absl::InternalError("No TFLitemodel found.");
}

absl::StatusOr<std::shared_ptr<SentencePieceTokenizer>>
ModelResources::GetTokenizer() {
  if (tokenizer_ != nullptr) {
    return tokenizer_;
  }
  // Lazily create the tokenizer from the task bundle or litertlm file.
  if (litert_lm_loader_ != nullptr) {
    ABSL_LOG(FATAL) << "Not supported file format in OSS yet.";
  } else if (model_asset_bundle_resources_ != nullptr) {
    ASSIGN_OR_RETURN(auto string_view,  // NOLINT
                     model_asset_bundle_resources_->GetFile("TOKENIZER_MODEL"));
    ASSIGN_OR_RETURN(auto tokenizer,  // NOLINT
                     SentencePieceTokenizer::CreateFromBuffer(string_view));
    tokenizer_ = std::move(tokenizer);
    return tokenizer_;
  }
  return absl::InternalError("No tokenizer found.");
}

}  // namespace litert::lm
