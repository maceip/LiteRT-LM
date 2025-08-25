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

#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/types.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

std::string GetTestdataPath(const std::string& file_name) {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir / file_name)
      .string();
}

absl::StatusOr<std::string> GetContents(const std::string& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(absl::StrCat("Could not open file: ", path));
  }

  std::string content;
  content.assign((std::istreambuf_iterator<char>(input_stream)),
                 (std::istreambuf_iterator<char>()));
  return std::move(content);
}

MATCHER_P(HasInputText, text_input, "") {
  return std::holds_alternative<InputText>(arg) &&
         std::get<InputText>(arg).GetData() == text_input.GetData();
}

MATCHER_P(HasInputImage, image_input, "") {
  if (!std::holds_alternative<InputImage>(arg)) {
    return false;
  }
  auto image_bytes = std::get<InputImage>(arg).GetRawImageBytes();
  if (!image_bytes.ok()) {
    return false;
  }
  return image_bytes.value() == image_input->GetRawImageBytes().value();
}

TEST(Gemma3DataProcessorTest, ToInputDataVectorTextOnly) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const std::string rendered_template_prompt =
      "<start_of_turn>user\ntest prompt\n<end_of_turn>";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content", "test prompt"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  EXPECT_THAT(input_data,
              ElementsAre(HasInputText(InputText(
                  "<start_of_turn>user\ntest prompt\n<end_of_turn>"))));
}

TEST(Gemma3DataProcessorTest, ToInputDataVectorTextAndImage) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const std::string rendered_template_prompt =
      "<start_of_turn>user\nHere is an image of a cat "
      "<start_of_image><end_of_turn>";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content",
       {{{"type", "text"}, {"text", "Here is an image of a cat"}},
        {{"type", "image"}}}}};
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));
  InputImage image_input("");
  EXPECT_THAT(
      input_data,
      ElementsAre(HasInputText(InputText(
                      "<start_of_turn>user\nHere is an image of a cat ")),
                  HasInputImage(&image_input),
                  HasInputText(InputText("<end_of_turn>"))));
}

TEST(Gemma3DataProcessorTest, ToMessage) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  Responses responses(1);
  responses.GetMutableResponseTexts()[0] = "test response";
  ASSERT_OK_AND_ASSIGN(const Message message,
                       processor->ToMessage(responses, std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(
      json_message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(Gemma3DataProcessorTest, PromptTemplateToInputDataVectorTextOnly) {
  const std::string test_file_path =
      GetTestdataPath("google-gemma-3-1b-it.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "Hello world!"}},
      {{"role", "user"}, {"content", "How are you?"}},
      {{"role", "assistant"},
       {"content", "I am doing well, thanks for asking."}},
      {{"role", "user"}, {"content", "What is the capital of France?"}},
  };
  PromptTemplateInput template_input = {.messages = messages,
                                        .add_generation_prompt = true};

  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_prompt, messages, {}));
  EXPECT_THAT(input_data,
              ElementsAre(HasInputText(InputText(R"""(<start_of_turn>user
Hello world!

How are you?<end_of_turn>
<start_of_turn>model
I am doing well, thanks for asking.<end_of_turn>
<start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
)"""))));
}

TEST(Gemma3DataProcessorTest, PromptTemplateToInputDataVectorTextAndImage) {
  const std::string test_file_path =
      GetTestdataPath("google-gemma-3-1b-it.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "Hello world!"}},
      {{"role", "user"},
       {"content",
        {{{"type", "text"}, {"text", "How are you?"}}, {{"type", "image"}}}}},
      {{"role", "assistant"},
       {"content", "I am doing well, thanks for asking."}},
      {{"role", "user"},
       {"content",
        {{{"type", "image"}},
         {{"type", "text"}, {"text", "What is the capital of France?"}}}}}};
  PromptTemplateInput template_input = {.messages = messages,
                                        .add_generation_prompt = true};

  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_prompt, messages, {}));
  InputImage image_input("");
  EXPECT_THAT(input_data,
              ElementsAre(HasInputText(InputText(R"""(<start_of_turn>user
Hello world!

How are you?)""")),
                          HasInputImage(&image_input),
                          HasInputText(InputText(R"""(<end_of_turn>
<start_of_turn>model
I am doing well, thanks for asking.<end_of_turn>
<start_of_turn>user
)""")),
                          HasInputImage(&image_input),
                          HasInputText(InputText(
                              R"""(What is the capital of France?<end_of_turn>
<start_of_turn>model
)"""))));
}

}  // namespace
}  // namespace litert::lm
