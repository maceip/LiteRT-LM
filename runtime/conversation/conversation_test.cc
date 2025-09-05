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

#include "runtime/conversation/conversation.h"

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

absl::string_view kTestLlmPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";

std::string GetTestdataPath(absl::string_view file_path) {
  return absl::StrCat(::testing::SrcDir(), "/", file_path);
}

class TestMessageObservable : public MessageObservable {
 public:
  explicit TestMessageObservable(const Message& expected_message)
      : expected_message_(expected_message) {}

  void OnError(const absl::Status& status) override {
    FAIL() << "OnError: " << status.message();
  }

  void OnMessage(const Message& message) override {
    const JsonMessage& json_message = std::get<JsonMessage>(message);
    JsonMessage& expected_json_message =
        std::get<JsonMessage>(expected_message_);
    // Compare the message text content by prefix, and update the expected
    // message to the remaining text for the next callback.
    ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
    ASSERT_TRUE(json_message["content"][0]["text"].is_string());
    std::string expected_string = expected_json_message["content"][0]["text"];
    std::string actual_string = json_message["content"][0]["text"];
    EXPECT_TRUE(absl::StartsWith(expected_string, actual_string))
        << "Expected: " << expected_string << "\nActual: " << actual_string;
    expected_json_message["content"][0]["text"] =
        expected_string.substr(actual_string.size());
  }

  void OnComplete() override {
    JsonMessage& expected_json_message =
        std::get<JsonMessage>(expected_message_);
    ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
    std::string expected_string = expected_json_message["content"][0]["text"];
    // The expected string should be empty after the last callback.
    EXPECT_TRUE(expected_string.empty());
    done_.Notify();
  }

  bool IsDone() { return done_.HasBeenNotified(); }

 private:
  Message expected_message_;
  absl::Notification done_;
};

TEST(ConversationTest, SendMessage) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session)));
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message = {
      {"role", "assistant"},
      {"content", {{{"type", "text"}, {"text", "하자ṅṅ zkušen помогут맷 ३"}}}}};
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  EXPECT_EQ(json_message, expected_message);
}

TEST(ConversationTest, SendMessageStream) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session)));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message = {
      {"role", "assistant"},
      {"content", {{{"type", "text"}, {"text", "하자ṅṅ zkušen помогут맷 ३"}}}}};
  TestMessageObservable test_observable(expected_message);
  EXPECT_OK(conversation->SendMessageStream(
      JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
      &test_observable));
  // Wait for the async message to be processed.
  while (!test_observable.IsDone()) {
    absl::SleepFor(absl::Milliseconds(100));
  }
}

TEST(ConversationTest, SendMessageWithPreface) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));
  Preface preface =
      JsonPreface{.messages = {{{"role", "system"},
                                {"content", "You are a helpful assistant."}}}};
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session), preface));
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message = {
      {"role", "assistant"},
      {"content",
       {{{"type", "text"}, {"text", "约为✗ predictions дыхаnewLineevu"}}}}};
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  EXPECT_EQ(json_message, expected_message);
}

}  // namespace
}  // namespace litert::lm
