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

#include "runtime/executor/magic_number_configs_helper.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

// Model without magic numbers.
//   prefill: context_length = 1280, prefill_length = 1024
//   decode: context_length = 1280
constexpr absl::string_view kTestModelPathNone = "magic_test_none.tflite";

// Model with magic numbers for context length.
//   prefill: context_length = 8209, prefill_length = 1024
//   decode: context_length = 8209
//   test_prefill_1280: context_length = 1280, prefill_length = 1024
//   test_decode_1280: context_length = 1280
constexpr absl::string_view kTestModelPathContextLength =
    "magic_test_context_length.tflite";

// Model with magic numbers both for context length and prefill length.
//   prefill: context_length = 8209, prefill_length = 4099
//   decode: context_length = 8209
//   test_prefill_1280: context_length = 1280, prefill_length = 1024
//   test_decode_1280: context_length = 1280
constexpr absl::string_view kTestModelPathBoth = "magic_test_both.tflite";

absl::StatusOr<Model> LoadModelFromFile(absl::string_view model_path) {
  auto model_path_in_srcdir =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/" /
      std::string(model_path);
  LITERT_ASSIGN_OR_RETURN(auto model,
                          Model::CreateFromFile(model_path_in_srcdir.string()));
  return model;
}

absl::StatusOr<LlmExecutorSettings> GetLlmExecutorSettings() {
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create("dont_care_path"));
  return LlmExecutorSettings::CreateDefault(std::move(model_assets));
}

TEST(MagicNumberConfigsHelperTest, None_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathNone);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  // No magic number configs and verifications.
  EXPECT_EQ(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_TRUE(env_options.empty());
}

TEST(MagicNumberConfigsHelperTest, None_ExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathNone);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_size = 1024};
  executor_settings->SetAdvancedSettings(advanced_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  // No magic number configs and verifications.
  EXPECT_EQ(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_TRUE(env_options.empty());
}

TEST(MagicNumberConfigsHelperTest, ContextLength_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 1);
}

TEST(MagicNumberConfigsHelperTest, ContextLength_ExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 1);
}

TEST(MagicNumberConfigsHelperTest,
     ContextLength_ExplictSettingsLargerThanMagicNumbers) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(9000);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 1);
}

TEST(MagicNumberConfigsHelperTest,
     ContextLength_ExplictSettingsWithVerifications) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 2);

  const auto& verification0 =
      helper.magic_number_verifications()->verifications[0];
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  const auto& verification1 =
      helper.magic_number_verifications()->verifications[1];
  EXPECT_EQ(std::string(verification1.signature), "prefill");
  EXPECT_EQ(std::string(verification1.test_signature), "test_prefill_1280");
  EXPECT_EQ(verification1.is_superset, true);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 2);
}

TEST(MagicNumberConfigsHelperTest, Both_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = helper.magic_number_configs()->configs[1];
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 4096);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 1);
}

TEST(MagicNumberConfigsHelperTest, Both_ExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_size = 1024};
  executor_settings->SetAdvancedSettings(advanced_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = helper.magic_number_configs()->configs[1];
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 1024);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 1);
}

TEST(MagicNumberConfigsHelperTest, Both_ExplictSettingsLargerThanMagicNumbers) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(9000);
  AdvancedSettings advanced_settings{.prefill_batch_size = 5000};
  executor_settings->SetAdvancedSettings(advanced_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = helper.magic_number_configs()->configs[1];
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 4096);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 1);
}

TEST(MagicNumberConfigsHelperTest, Both_ExplictSettingsWithVerifications) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_size = 1024,
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = helper.magic_number_configs()->configs[1];
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 1024);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 2);

  const auto& verification0 =
      helper.magic_number_verifications()->verifications[0];
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  const auto& verification1 =
      helper.magic_number_verifications()->verifications[1];
  EXPECT_EQ(std::string(verification1.signature), "prefill");
  EXPECT_EQ(std::string(verification1.test_signature), "test_prefill_1280");
  EXPECT_EQ(verification1.is_superset, true);

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 2);
}

TEST(MagicNumberConfigsHelperTest,
     Both_ExplictSettingsWithVerifications_MatchedPartially) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_size = 512,  // Not matched.
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  MagicNumberConfigsHelper helper(*model, *executor_settings);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = helper.magic_number_configs()->configs[0];
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = helper.magic_number_configs()->configs[1];
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 512);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 1);

  const auto& verification0 =
      helper.magic_number_verifications()->verifications[0];
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  // prefill won't be verified because prefill_batch_size is not matched with
  // test_prefill_1280.

  auto env_options = helper.GetLiteRtEnvOptions();
  EXPECT_EQ(env_options.size(), 2);
}

}  // namespace
}  // namespace litert::lm
