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

#include "runtime/components/model_resources_litert_lm.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesLitertLm::Create(
    std::unique_ptr<LitertLmLoader> litert_lm_loader) {
  return absl::WrapUnique(
      new ModelResourcesLitertLm(std::move(litert_lm_loader)));
};

absl::StatusOr<std::shared_ptr<litert::Model>>
ModelResourcesLitertLm::GetTFLiteModel(ModelType model_type) {
  if (model_ != nullptr) {
    return model_;
  }
  litert::BufferRef<uint8_t> buffer_ref =
      litert_lm_loader_->GetTFLiteModel(model_type);
  ABSL_LOG(INFO) << "litert model size: " << buffer_ref.Size();
  // TODO: b/413214239 - This factory function copies the contents of
  // `buffer_ref`. Ideally we'd create a `Model` backed by a view of mapped
  // memory.
  LITERT_ASSIGN_OR_RETURN(auto model, Model::CreateFromBuffer(buffer_ref));
  model_ = std::make_shared<litert::Model>(std::move(model));
  return model_;
}

absl::StatusOr<std::shared_ptr<SentencePieceTokenizer>>
ModelResourcesLitertLm::GetTokenizer() {
  if (tokenizer_ != nullptr) {
    return tokenizer_;
  }
  auto buffer_ref = litert_lm_loader_->GetTokenizer();
  ASSIGN_OR_RETURN(  // NOLINT
      auto tokenizer,
      SentencePieceTokenizer::CreateFromBuffer(buffer_ref.StrView()));
  tokenizer_ = (std::move(tokenizer));
  return tokenizer_;
}

absl::StatusOr<std::shared_ptr<proto::LlmMetadata>>
ModelResourcesLitertLm::GetLlmMetadata() {
  if (llm_metadata_ != nullptr) {
    return llm_metadata_;
  }
  auto buffer_ref = litert_lm_loader_->GetLlmMetadata();
  llm_metadata_ = std::make_shared<proto::LlmMetadata>();
  llm_metadata_->ParseFromString(std::string(buffer_ref.StrView()));  // NOLINT
  return llm_metadata_;
};

}  // namespace litert::lm
