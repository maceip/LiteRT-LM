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
#include <utility>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"

namespace litert::lm {

// ModelResources is a wrapper around the .litertlm or .task file format.
// First, it manages all the loaded model resources that need to be hold to
// avoid the model being destroyed. So this object lifetime need to be longer
// than the models it manages, like the tokenizer and the litert model.
// Second, it provides a way to load the models from either file format in a
// lazy way. Basically, it will create the models when they are actually used.
// Before the Get*() functions are called, the models are not created yet. And
// once the models are created, they will be re-used for all the following
// calls.
// Third, it provides a way to memory map the models from either file format.
class ModelResources {
 public:
  // Creates a ModelResources from a LitertLmLoader for a .litertlm file.
  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      std::unique_ptr<LitertLmLoader> litert_lm_loader) {
    return absl::WrapUnique(new ModelResources(std::move(litert_lm_loader)));
  };

  // Creates a ModelResources from a ModelAssetBundleResources, which is from a
  // .task file.
  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources) {
    return absl::WrapUnique(
        new ModelResources(std::move(model_asset_bundle_resources)));
  };

  // Deprecated. Creates a ModelResources from a litert::Model.
  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      std::unique_ptr<litert::Model> model) {
    return absl::WrapUnique(new ModelResources(std::move(model)));
  };

  // Returns the litert model. We will create the model if it is not created
  // yet. And the model is created from memory mapped file, so physical memory
  // is only allocated when the model is actually used.
  absl::StatusOr<std::shared_ptr<litert::Model>> GetTFLiteModel();

  // Returns the tokenizer.
  absl::StatusOr<std::shared_ptr<SentencePieceTokenizer>> GetTokenizer();

 private:
  // Constructor using a LitertLmLoader.
  explicit ModelResources(std::unique_ptr<LitertLmLoader> litert_lm_loader)
      : litert_lm_loader_(std::move(litert_lm_loader)) {};
  // Constructor using a ModelAssetBundleResources.
  explicit ModelResources(
      std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources)
      : model_asset_bundle_resources_(std::move(model_asset_bundle_resources)) {
        };
  // Depreated.Constructor using a litert::Model.
  explicit ModelResources(std::unique_ptr<litert::Model> model)
      : litert_model_(std::move(model)) {};

  // The litert model.
  std::shared_ptr<::litert::Model> litert_model_;
  // The tokenizer.
  std::shared_ptr<SentencePieceTokenizer> tokenizer_;

  // The model asset bundle resources produced by reading task bundle. Not null
  // only when the model is provided through .task format. If the model is
  // retrieved from this resource, releasing this resource will also invalidate
  // the model.
  std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources_;
  // The litert lm loader, used to mmap the tokenizer and tflite model etc from
  // the .litertlm model file.
  std::unique_ptr<LitertLmLoader> litert_lm_loader_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
