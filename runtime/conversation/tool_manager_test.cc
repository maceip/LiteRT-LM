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

#include "runtime/conversation/tool_manager.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::Eq;

// Fake implementation of ToolManager for testing.
class FakeToolManager : public ToolManager {
 public:
  absl::StatusOr<JsonMessage> GetToolsDescription() override {
    return JsonMessage::parse(R"([
      {
        "name": "addIntegers",
        "description": "Adds two integers and returns the result.",
        "parameters": {
          "type": "object",
          "properties": {
            "first": {
              "description": "The first integer.",
              "type": "integer",
              "nullable": false
            },
            "second": {
              "description": "The second integer.",
              "type": "integer",
              "nullable": false
            }
          },
          "required": [
            "first",
            "second"
          ]
        }
      },
      {
        "name": "getWeather",
        "description": "Gets the weather for a given city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "description": "City Name",
              "type": "string",
              "nullable": false
            },
            "country": {
              "description": "Country Name",
              "type": "string",
              "nullable": true
            }
          },
          "required": [
            "city"
          ]
        }
      }
    ])");
  }

  absl::StatusOr<JsonMessage> Execute(const std::string& functionName,
                                      const JsonMessage& params) override {
    if (functionName == "addIntegers") {
      if (params.contains("first") && params.contains("second") &&
          params["first"].is_number_integer() &&
          params["second"].is_number_integer()) {
        int first = params["first"];
        int second = params["second"];
        return JsonMessage{{"result", first + second}};
      } else {
        return JsonMessage{{"error", "Invalid parameters for addIntegers"}};
      }
    } else if (functionName == "getWeather") {
      if (params.contains("city") && params["city"].is_string()) {
        std::string city = params["city"];
        return JsonMessage{{"weather", "Sunny in " + city}};
      } else {
        return JsonMessage{{"error", "Invalid parameters for getWeather"}};
      }
    }
    return JsonMessage{{"error", "Function not found"}};
  }
};

TEST(FakeToolManagerTest, GetToolsDescription) {
  FakeToolManager tool_manager;
  ASSERT_OK_AND_ASSIGN(JsonMessage spec, tool_manager.GetToolsDescription());
  ASSERT_TRUE(spec.is_array());
  ASSERT_EQ(spec.size(), 2);

  EXPECT_THAT(spec[0]["name"], Eq("addIntegers"));
  EXPECT_THAT(spec[0]["description"],
              Eq("Adds two integers and returns the result."));

  EXPECT_THAT(spec[1]["name"], Eq("getWeather"));
  EXPECT_THAT(spec[1]["description"], Eq("Gets the weather for a given city."));
}

TEST(FakeToolManagerTest, ExecuteAddIntegers) {
  FakeToolManager tool_manager;
  JsonMessage params = {{"first", 10}, {"second", 20}};
  ASSERT_OK_AND_ASSIGN(JsonMessage result,
                       tool_manager.Execute("addIntegers", params));
  EXPECT_THAT(result["result"], Eq(30));
}

TEST(FakeToolManagerTest, ExecuteAddIntegersInvalidParams) {
  FakeToolManager tool_manager;
  JsonMessage params = {{"first", 10}};
  ASSERT_OK_AND_ASSIGN(JsonMessage result,
                       tool_manager.Execute("addIntegers", params));
  EXPECT_THAT(result["error"], Eq("Invalid parameters for addIntegers"));
}

TEST(FakeToolManagerTest, ExecuteGetWeather) {
  FakeToolManager tool_manager;
  JsonMessage params = {{"city", "London"}};
  ASSERT_OK_AND_ASSIGN(JsonMessage result,
                       tool_manager.Execute("getWeather", params));
  EXPECT_THAT(result["weather"], Eq("Sunny in London"));
}

TEST(FakeToolManagerTest, ExecuteGetWeatherInvalidParams) {
  FakeToolManager tool_manager;
  JsonMessage params = {};
  ASSERT_OK_AND_ASSIGN(JsonMessage result,
                       tool_manager.Execute("getWeather", params));
  EXPECT_THAT(result["error"], Eq("Invalid parameters for getWeather"));
}

TEST(FakeToolManagerTest, ExecuteUnknownFunction) {
  FakeToolManager tool_manager;
  JsonMessage params = {};
  ASSERT_OK_AND_ASSIGN(JsonMessage result,
                       tool_manager.Execute("unknown_function", params));
  EXPECT_THAT(result["error"], Eq("Function not found"));
}

}  // namespace
}  // namespace litert::lm
