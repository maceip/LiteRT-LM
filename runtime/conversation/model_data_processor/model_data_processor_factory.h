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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_FACTORY_H_

#include <memory>
#include <optional>
#include <variant>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/proto/llm_model_type.pb.h"

namespace litert::lm {

// Creates a ModelDataProcessor instance based on the given model type and
// config.
absl::StatusOr<std::unique_ptr<ModelDataProcessor>> CreateModelDataProcessor(
    const proto::LlmModelType& llm_model_type,
    const DataProcessorConfig& config = std::monostate(),
    std::optional<Preface> preface = std::nullopt);

// Returns the default data processor config for the given model type.
DataProcessorConfig GetDefaultDataProcessorConfig(
    const proto::LlmModelType& llm_model_type);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_FACTORY_H_
