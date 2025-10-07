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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_TOOL_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_TOOL_MANAGER_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"

namespace litert::lm {

// Interface for managing and executing tools.
//
// This interface is to enable function calls (also known as tool use) in the
// Conversation API.
class ToolManager {
 public:
  virtual ~ToolManager() = default;

  // Get a json array of the tools description in OpenAPI format.
  //
  // Each array element represents the description of a single tool.
  virtual absl::StatusOr<JsonMessage> GetToolsDescription() = 0;

  // Execute a tool and returns a json object as result.
  //
  // The format is very free as long as it is a json object understandable by
  // the LLM.
  //
  // For example,
  // - return multiple fields (`{"temperature": 20, "unit": "celsius", weather:
  // "sunny"}`) to give better context.
  // - return a simple "okay" (`{"result": "okay"}`) if the action is done
  // without further info.
  // - return a string of error message when error happens: `{"error": "failed
  // to execute"}`.
  virtual absl::StatusOr<JsonMessage> Execute(const std::string& functionName,
                                              const JsonMessage& params) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_TOOL_MANAGER_H_
