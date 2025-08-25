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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_ARGUMENTS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_ARGUMENTS_H_

#include <string>

namespace litert::lm {

// Config for Gemma3DataProcessor.
// TODO: b/438830175, b/436674053 - Refactor these config to Image Preprocessor
// Configs once image preprocessor is ready.
struct Gemma3DataProcessorConfig {
  // The number of image soft tokens for a single image.
  int num_image_tokens = 256;
  // The string for beginning of image token.
  std::string boi_token = "<start_of_image>";
  // The string for image soft token.
  std::string image_token = "<image_soft_token>";
  // The string for end of image token.
  std::string eoi_token = "<end_of_image>";
};

// Arguments for Gemma3DataProcessor.
struct Gemma3DataProcessorArguments {};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_ARGUMENTS_H_
