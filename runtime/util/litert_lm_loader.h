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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {

// A class to load the Litert LM model from the .litertlm file. The loader will
// read the model header from and map the sections to the section buffers.
class LitertLmLoader {
 public:
  // Creates a LitertLmLoader from the model file. The loader will read the
  // model header from and map the sections to the section buffers.
  explicit LitertLmLoader(ScopedFile model_file)
      : model_file_(std::move(model_file)) {
    ABSL_CHECK_OK(Initialize());
  }
  // Returns the tokenizer section buffer.
  litert::BufferRef<uint8_t> GetTokenizer() {
    return section_buffers_[litertlm::schema::AnySectionDataType_SP_Tokenizer];
  }

  // Returns the TFLite model section buffer.
  litert::BufferRef<uint8_t> GetTFLiteModel() {
    return section_buffers_[litertlm::schema::AnySectionDataType_TFLiteModel];
  };

  // Returns the tokenizer section buffer.
  litert::BufferRef<uint8_t> GetLlmMetadata() {
    return section_buffers_
        [litertlm::schema::AnySectionDataType_LlmMetadataProto];
  }

 private:
  // Initializes the LitertLmLoader. Includes reading the model header and
  // mapping the sections to the section buffers.
  absl::Status Initialize();
  // Maps the sections to the section buffers.
  absl::Status MapSections();

  // The model file to be loaded.
  ScopedFile model_file_;
  // The model_file_ mapped to a MemoryMappedFile.
  std::unique_ptr<MemoryMappedFile> memory_mapped_file_;

  // TODO (b/413793273): Add the extra names to the key to differentiate
  // between the TFLite models.
  std::unordered_map<litertlm::schema::AnySectionDataType, BufferRef<uint8_t>>
      section_buffers_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_
