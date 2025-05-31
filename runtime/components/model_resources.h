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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_

// All the loaded model resources the executor needs to hold to avoid the model
// being destroyed.
#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

// ModelResources is an interface that manages all the loaded model resources
// that need to be hold to avoid the model being destroyed. So this object's
// lifetime need to be longer than the models it manages, like the tokenizer
// and the litert model. It provides a way to load the models in a lazy way.
// Basically, it will create the models when they are actually used. Before the
// Get*() functions are called, the models are not created yet. And once the
// models are created, they will be re-used for all the following calls.
class ModelResources {
 public:
  virtual ~ModelResources() = default;

  // Returns the litert model. We will create the model if it is not created
  // yet. And the model is created from memory mapped file, so physical memory
  // is only allocated when the model is actually used.
  // TODO: b/413214239 - Load the model from mapped memory without creating an
  // extra copy.
  virtual absl::StatusOr<std::shared_ptr<litert::Model>> GetTFLiteModel() = 0;

  // Returns the tokenizer.
  virtual absl::StatusOr<std::shared_ptr<SentencePieceTokenizer>>
  GetTokenizer() = 0;

  // Returns the llm metadata.
  virtual absl::StatusOr<std::shared_ptr<proto::LlmMetadata>>
  GetLlmMetadata() = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
