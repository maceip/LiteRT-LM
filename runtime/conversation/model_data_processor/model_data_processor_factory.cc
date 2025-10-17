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

#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"

#include <memory>
#include <optional>
#include <variant>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/proto/llm_model_type.pb.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> CreateModelDataProcessor(
    const proto::LlmModelType& model_type, const Tokenizer& tokenizer,
    const DataProcessorConfig& config, std::optional<Preface> preface) {
  switch (model_type.model_type_case()) {
    case proto::LlmModelType::kGemma3N:
    case proto::LlmModelType::kGemma3:
      ABSL_LOG(INFO) << "Creating Gemma3DataProcessor for model type: "
                     << model_type.model_type_case();
      return Gemma3DataProcessor::Create(
          std::holds_alternative<Gemma3DataProcessorConfig>(config)
              ? std::get<Gemma3DataProcessorConfig>(config)
              : Gemma3DataProcessorConfig(),
          preface);
    case proto::LlmModelType::kQwen3:
      ABSL_LOG(INFO) << "Creating Qwen3DataProcessor for model type: "
                     << model_type.model_type_case();
      return Qwen3DataProcessor::Create(
          std::holds_alternative<Qwen3DataProcessorConfig>(config)
              ? std::get<Qwen3DataProcessorConfig>(config)
              : Qwen3DataProcessorConfig());
    case proto::LlmModelType::kGenericModel: {
      ABSL_LOG(INFO) << "Creating GenericDataProcessor for model type: "
                     << model_type.model_type_case();
      if (std::holds_alternative<GenericDataProcessorConfig>(config)) {
        return GenericDataProcessor::Create(
            std::holds_alternative<GenericDataProcessorConfig>(config)
                ? std::get<GenericDataProcessorConfig>(config)
                : GenericDataProcessorConfig());
      } else {
        return GenericDataProcessor::Create();
      }
    }
    default:
      return absl::InvalidArgumentError("Unsupported model type");
  }
  return absl::InvalidArgumentError("Unsupported model type");
}

DataProcessorConfig GetDefaultDataProcessorConfig(
    const proto::LlmModelType& llm_model_type) {
  switch (llm_model_type.model_type_case()) {
    case proto::LlmModelType::kGemma3N:
    case proto::LlmModelType::kGemma3:
      return Gemma3DataProcessorConfig();
    case proto::LlmModelType::kQwen3:
      return Qwen3DataProcessorConfig();
    case proto::LlmModelType::kGenericModel:
      return GenericDataProcessorConfig();
    default:
      return std::monostate();
  }
}

}  // namespace litert::lm
