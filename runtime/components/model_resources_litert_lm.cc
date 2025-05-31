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
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
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
ModelResourcesLitertLm::GetTFLiteModel() {
  if (model_ != nullptr) {
    return model_;
  }
  ABSL_LOG(FATAL) << "Not supported file format in OSS yet.";
}

absl::StatusOr<std::shared_ptr<SentencePieceTokenizer>>
ModelResourcesLitertLm::GetTokenizer() {
  if (tokenizer_ != nullptr) {
    return tokenizer_;
  }
  ABSL_LOG(FATAL) << "Not supported file format in OSS yet.";
}

absl::StatusOr<std::shared_ptr<proto::LlmMetadata>>
ModelResourcesLitertLm::GetLlmMetadata() {
  if (llm_metadata_ != nullptr) {
    return llm_metadata_;
  }
  ABSL_LOG(FATAL) << "Not supported file format in OSS yet.";
};

}  // namespace litert::lm
