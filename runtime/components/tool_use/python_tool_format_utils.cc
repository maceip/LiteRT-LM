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

#include "runtime/components/tool_use/python_tool_format_utils.h"

#include <sstream>
#include <string>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {
namespace {

std::string FormatParameterType(absl::string_view key,
                                const nlohmann::ordered_json& schema,
                                bool is_required) {
  std::stringstream ss;
  std::string type = schema.value("type", "");

  if (type == "boolean") {
    ss << "bool";
  } else if (type == "integer") {
    ss << "int";
  } else if (type == "number") {
    ss << "float";
  } else if (type == "string") {
    ss << "str";
  } else if (type == "array") {
    if (schema.contains("items") && schema["items"].is_object()) {
      ss << "list[" << FormatParameterType(key, schema["items"], true) << "]";
    } else {
      ss << "list[Any]";
    }
  } else if (type == "object") {
    ss << "dict";
  } else {
    ss << "Any";
  }

  if (!is_required) {
    ss << " | None = None";
  }

  return ss.str();
}

std::string GenerateDocstring(const nlohmann::ordered_json& tool) {
  std::stringstream ss;

  if (tool.contains("description")) {
    ss << tool["description"].get<std::string>() << "\n";
  }

  // Generate argument descriptions.
  if (tool.contains("parameters") &&
      tool["parameters"].contains("properties")) {
    ss << "\n  Args:\n";
    for (const auto& [key, value] : tool["parameters"]["properties"].items()) {
      ss << "    " << key;

      if (value.contains("description")) {
        ss << ": " << value["description"].get<std::string>() << "\n";
      }
    }
  }

  return ss.str();
}

}  // namespace

absl::StatusOr<std::string> FormatToolAsPython(
    const nlohmann::ordered_json& tool) {
  if (!tool.contains("name")) {
    return absl::InvalidArgumentError("Tool name is required.");
  }

  std::stringstream ss;
  ss << "def " << tool["name"].get<std::string>() << "(";

  if (tool.contains("parameters") &&
      tool["parameters"].contains("properties")) {
    ss << "\n";
    const nlohmann::ordered_json required_params =
        tool["parameters"].value("required", nlohmann::ordered_json::array());
    absl::flat_hash_set<std::string> required(required_params.begin(),
                                              required_params.end());
    int count = 0;
    for (const auto& [key, value] : tool["parameters"]["properties"].items()) {
      const bool is_required = required.contains(key);
      ss << "    " << key << ": ";
      ss << FormatParameterType(key, value, is_required);
      ss << ",";
      if (++count < tool["parameters"]["properties"].size()) {
        ss << "\n";
      }
    }
    ss << "\n";
  }

  ss << ") -> dict:\n";

  std::string docstring = GenerateDocstring(tool);
  if (!docstring.empty()) {
    ss << "  \"\"\"";
    ss << docstring;
    ss << "  \"\"\"\n";
  }

  return ss.str();
}

}  // namespace litert::lm
