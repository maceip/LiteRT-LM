// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

struct GpuArtisanConfig {
  // Number of output candidates.
  uint32_t num_output_candidates = 1;

  // Whether to wait for weight uploads before prefilling.
  bool wait_for_weight_uploads = false;

  // Number of decode steps per sync. Used by GPU only.
  uint32_t num_decode_steps_per_sync = 1;

  // Sequence batch size for encoding. Used by GPU only. Number of input
  // tokens to process at a time for batch processing. Setting this value to 1
  // means both the encoding and decoding share the same graph of sequence
  // length of 1. Setting this value to 0 means the batch size will be
  // optimized programmatically.
  uint32_t sequence_batch_size = 0;

  // The supported lora ranks for the base model. Used by GPU only. By default
  // it will be empty, meaning not supporting any lora ranks.
  std::vector<uint32_t> supported_lora_ranks = {};

  // Maximum top k, which is the max Top-K value supported for all
  // sessions created with the engine, used by GPU only. If a session with
  // Top-K value larger than this is being asked to be created, it will be
  // rejected(throw error). The max top k will be 1, which means only greedy
  // decoding is supported for any sessions created with this engine.
  uint32_t max_top_k = 1;

  // Enables decode logits.
  // AiCore uses decode logits, so this is enabled for AiCore.
  // LLM Engine defaults to disabling decode logits.
  bool enable_decode_logits = false;
};

std::ostream& operator<<(std::ostream& os, const GpuArtisanConfig& config);

struct GpuConfig {
  // Maximum top k, which is the max Top-K value supported for all
  // sessions created with the engine, used by GPU only. If a session with
  // Top-K value larger than this is being asked to be created, it will be
  // rejected(throw error). The default max top k will be 1, which
  // means only greedy decoding is supported for any sessions created with
  // this engine.
  uint32_t max_top_k = 1;
};
std::ostream& operator<<(std::ostream& os, const GpuConfig& config);

struct CpuConfig {
  // Number of threads. The default value is 4.
  uint32_t number_of_threads = 4;
};
std::ostream& operator<<(std::ostream& os, const CpuConfig& config);

// Settings for the LLM executor.
//
// This class holds the settings for the LLM executor, including the
// model assets, cache directory, maximum number of tokens, backend,
// activation data type, and backend-specific settings.
//
// The user should construct the class using ModelAssets and then set the
// remaining settings using the setter APIs.
class LlmExecutorSettings : public ExecutorSettingsBase {
 public:
  // Creates a LlmExecutorSettings with default values using the provided
  // ModelAssets.
  static absl::StatusOr<LlmExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, Backend backend = Backend::CPU);

  // Getter APIs.
  uint32_t GetMaxNumTokens() const { return max_num_tokens_; }
  uint32_t GetMaxNumImages() const { return max_num_images_; }

  template <typename T>
  absl::StatusOr<const T> GetBackendConfig() const {
    if (std::holds_alternative<T>(backend_config_)) {
      return std::get<T>(backend_config_);
    } else {
      return absl::InvalidArgumentError("Backend config is not valid.");
    }
  }

  template <typename T>
  absl::StatusOr<T> MutableBackendConfig() {
    if (std::holds_alternative<T>(backend_config_)) {
      return std::get<T>(backend_config_);
    } else {
      return absl::InvalidArgumentError("Backend config is not valid.");
    }
  }

  void SetMaxNumTokens(uint64_t max_num_tokens) {
    max_num_tokens_ = max_num_tokens;
  }
  void SetMaxNumImages(uint32_t max_num_images) {
    max_num_images_ = max_num_images;
  }

  void SetBackendConfig(const std::variant<GpuArtisanConfig, GpuConfig,
                                           CpuConfig>& backend_config) {
    backend_config_ = backend_config;
  }

 private:
  explicit LlmExecutorSettings(const ModelAssets& model_assets)
      : ExecutorSettingsBase(model_assets) {}

  // Maximum number of the sum of input and output tokens. It is equivalent to
  // the size of the kv-cache.
  uint32_t max_num_tokens_;

  // Maximum number of images the model can handle.
  uint32_t max_num_images_;

  // Backend specific config.
  std::variant<GpuArtisanConfig, GpuConfig, CpuConfig> backend_config_;

  // Declare the output stream operator as a friend such that it can be used
  // to print the LlmExecutorSettings private member.
  friend std::ostream& operator<<(std::ostream& os,
                                  const LlmExecutorSettings& config);
};
std::ostream& operator<<(std::ostream& os, const LlmExecutorSettings& config);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_LLM_EXECUTOR_SETTINGS_H_
