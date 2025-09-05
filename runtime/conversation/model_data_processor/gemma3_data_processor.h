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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_DATA_PROCESSOR_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Gemma3DataProcessor is a model data processor for Gemma3 models.
class Gemma3DataProcessor
    : public TypeSafeModelDataProcessor<Gemma3DataProcessorConfig,
                                        Gemma3DataProcessorArguments> {
 public:
  // Creates a Gemma3DataProcessor instance.
  static absl::StatusOr<std::unique_ptr<Gemma3DataProcessor>> Create(
      Gemma3DataProcessorConfig config = Gemma3DataProcessorConfig());

  // Returns the config of the Gemma3DataProcessor.
  const Gemma3DataProcessorConfig& GetConfig() override { return config_; }

 private:
  explicit Gemma3DataProcessor(
      const Gemma3DataProcessorConfig& config = Gemma3DataProcessorConfig())
      : config_(config) {};
  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const Gemma3DataProcessorArguments& args) override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const Gemma3DataProcessorArguments& args) override;

  Gemma3DataProcessorConfig config_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_GEMMA3_DATA_PROCESSOR_H_
