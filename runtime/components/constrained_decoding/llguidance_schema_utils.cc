// Copyright 2026 The ODML Authors.
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

#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

absl::StatusOr<std::string> FormatToolsAsLarkGrammar(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options) {
  std::vector<std::string> tool_names;
  for (const auto& tool : tools) {
    if (tool.contains("name") && tool["name"].is_string()) {
      tool_names.push_back(tool["name"].get<std::string>());
    }
  }
  std::string tool_union =
      absl::StrFormat(R"(TOOL_UNION: /%s/)", absl::StrJoin(tool_names, "|"));

  // Syntax to ensure it's a valid JSON with string escapes. But it doesn't
  // constrain fields against tools schema.
  std::string json_grammar =
      absl::StrFormat(R"(
fc_esc_open: %s
fc_esc_close: %s

key: IDENTIFIER
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
json_value: custom_string | NUMBER | BOOLEAN | NULL | object | array

custom_string: fc_esc_open /(.|\n)*/ fc_esc_close
array: "[" [json_value ("," json_value)*] "]"
object: "{" [pair ("," pair)*] "}"
pair: key ":" json_value

// Primitives (Standard JSON)
NUMBER: /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/
BOOLEAN: "true" | "false"
NULL: "null"
%%ignore /[ \t\r\n]+/)",
                      options.fc_open_quote, options.fc_close_quote);
  // Function calling syntax.
  std::string function_block = absl::StrFormat(
      R"((fc_start "call:" TOOL_UNION object fc_end)+ fc_resp
fc_start: %s
fc_end: %s
fc_resp: %s
)",
      options.fc_code_fence_start, options.fc_code_fence_end,
      options.fc_function_response_start);
  // Text only syntax. Special tokens are disallowed in terminals.
  std::string text_only_block = absl::StrFormat(
      R"(
FORBIDDEN_CALL : /.*%s.*/
SAFE_TEXT : /(.|\n)*/ & ~FORBIDDEN_CALL
start : SAFE_TEXT
)",
      options.fc_code_fence_start);

  std::string start_rule;
  switch (options.constraint_mode) {
    case LlgConstraintMode::kTextOnly: {
      return text_only_block;
    }
    case LlgConstraintMode::kFunctionCallsOnly: {
      if (tool_names.empty()) {
        return absl::InvalidArgumentError(
            "No tools provided for FunctionCallsOnly mode.");
      }
      start_rule = absl::StrCat("start: ", function_block, "\n");
      break;
    }
    case LlgConstraintMode::kTextAndOrFunctionCalls: {
      if (tool_names.empty()) {
        return text_only_block;
      }
      std::string tool_names_regex = absl::StrJoin(tool_names, "|");
      start_rule = absl::StrFormat(
          R"(
start: TEXT_CONTENT? function_block_opt
TEXT_CONTENT: /(.|\n)+/
function_block_opt: function_block |
function_block: %s
)",
          function_block);
      break;
    }
  }

  return absl::StrCat(tool_union, "\n", json_grammar, "\n", start_rule);
}

}  // namespace litert::lm
