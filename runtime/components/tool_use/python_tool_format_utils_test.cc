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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace {

using ::litert::lm::FormatToolAsPython;
using ::testing::status::IsOkAndHolds;

TEST(PythonToolFormatUtilsTest, FormatToolWithStringParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithMultipleParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          },
          "test_param_2": {
            "type": "string",
            "description": "Second parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str | None = None,
    test_param_2: str | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
    test_param_2: Second parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithRequiredParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          },
          "test_param_2": {
            "type": "string",
            "description": "Second parameter."
          }
        },
        "required": ["test_param_1"]
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str,
    test_param_2: str | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
    test_param_2: Second parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithArrayParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "First parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: list[str] | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithObjectParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "object",
            "properties": {
              "field_1": {
                "type": "string"
              }
            },
            "description": "First parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: dict | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithMixedParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          },
          "test_param_2": {
            "type": "object",
            "properties": {
              "field_1": {
                "type": "string"
              }
            },
            "description": "Second parameter."
          },
          "test_param_3": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "Third parameter."
          }
        },
        "required": ["test_param_1", "test_param_2", "test_param_3"]
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str,
    test_param_2: dict,
    test_param_3: list[str],
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
    test_param_2: Second parameter.
    test_param_3: Third parameter.
  """
)"));
}

}  // namespace
