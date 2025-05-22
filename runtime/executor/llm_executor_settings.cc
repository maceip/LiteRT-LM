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

#include "runtime/executor/llm_executor_settings.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/util/logging.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

std::ostream& operator<<(std::ostream& os, const Backend& backend) {
  switch (backend) {
    case Backend::CPU_ARTISAN:
      return os << "CPU_ARTISAN";
    case Backend::GPU_ARTISAN:
      return os << "GPU_ARTISAN";
    case Backend::GPU:
      return os << "GPU";
    case Backend::CPU:
      return os << "CPU";
    case Backend::GOOGLE_TENSOR_ARTISAN:
      return os << "GOOGLE_TENSOR_ARTISAN";
    default:
      return os << "UNKNOWN";
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ActivationDataType& activation) {
  switch (activation) {
    case ActivationDataType::FLOAT32:
      return os << "FLOAT32";
    case ActivationDataType::FLOAT16:
      return os << "FLOAT16";
    case ActivationDataType::INT16:
      return os << "INT16";
    case ActivationDataType::INT8:
      return os << "INT8";
    default:
      return os << "UNKNOWN";
  }
}


std::ostream& operator<<(std::ostream& os,
                         const FakeWeightsMode& fake_weights_mode) {
  switch (fake_weights_mode) {
    case FakeWeightsMode::FAKE_WEIGHTS_NONE:
      return os << "FAKE_WEIGHTS_NONE";
    case FakeWeightsMode::FAKE_WEIGHTS_8BITS_ALL_LAYERS:
      return os << "FAKE_WEIGHTS_8BITS_ALL_LAYERS";
    case FakeWeightsMode::FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4:
      return os << "FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4";
    default:
      return os << "FAKE_WEIGHTS_NONE";
  }
}

std::ostream& operator<<(std::ostream& os, const FileFormat& file_format) {
  switch (file_format) {
    case FileFormat::TFLITE:
      return os << "TFLITE";
    case FileFormat::TASK:
      return os << "TASK";
    case FileFormat::LITERT_LM:
      return os << "LITERT_LM";
  }
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(absl::string_view model_path) {
  return ModelAssets(std::string(model_path));
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::ScopedFile> model_file) {
  return ModelAssets(std::move(model_file));
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::ScopedFile> model_file,
    absl::string_view model_path) {
  if (model_file) {
    return Create(std::move(model_file));
  }
  return Create(model_path);
}

ModelAssets::ModelAssets(std::shared_ptr<litert::lm::ScopedFile> model_file)
    : path_or_scoped_file_(std::move(model_file)) {}

ModelAssets::ModelAssets(std::string model_path)
    : path_or_scoped_file_(std::move(model_path)) {}

std::ostream& operator<<(std::ostream& os, const ModelAssets& model_assets) {
  if (model_assets.HasScopedFile()) {
    os << "model_file file descriptor ID: "
       << model_assets.GetScopedFile().value()->file() << "\n";
  } else {
    os << "model_path: " << model_assets.GetPath().value() << "\n";
  }
  os << "fake_weights_mode: " << model_assets.fake_weights_mode() << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const GpuArtisanConfig& config) {
  os << "num_output_candidates: " << config.num_output_candidates << "\n";
  os << "wait_for_weight_uploads: " << config.wait_for_weight_uploads << "\n";
  os << "num_decode_steps_per_sync: " << config.num_decode_steps_per_sync
     << "\n";
  os << "sequence_batch_size: " << config.sequence_batch_size << "\n";
  os << "supported_lora_ranks: " << config.supported_lora_ranks << "\n";
  os << "max_top_k: " << config.max_top_k << "\n";
  os << "enable_decode_logits: " << config.enable_decode_logits << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const GpuConfig& config) {
  os << "max_top_k: " << config.max_top_k << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const CpuConfig& config) {
  os << "number_of_threads: " << config.number_of_threads << "\n";
  return os;
}

absl::StatusOr<
    std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
LlmExecutorSettings::GetWeightCacheFile(absl::string_view suffix) const {
  // Cache is explicitly disabled.
  if (GetCacheDir() == ":nocache") {
    return absl::InvalidArgumentError("Cache is explicitly disabled.");
  }

  // Prefer to use the scoped cache file if it's set.
  if (GetScopedCacheFile()) {
    return GetScopedCacheFile();
  }

  auto model_path = GetModelAssets().GetPath().value_or("");

  // There is no model path to suffix.
  if (model_path.empty()) {
    return absl::InvalidArgumentError(
        "Cache path cannot be computed without knowing the model path.");
  }

  if (GetCacheDir().empty()) {
    return absl::StrCat(model_path, suffix);
  }

  return JoinPath(GetCacheDir(), absl::StrCat(Basename(model_path), suffix));
}

std::ostream& operator<<(std::ostream& os, const LlmExecutorSettings& config) {
  os << "backend: " << config.GetBackend() << "\n";
  std::visit(
      [&os](const auto& backend_config) {
        os << "backend_config: " << backend_config << "\n";
      },
      config.backend_config_);
  os << "max_tokens: " << config.GetMaxNumTokens() << "\n";
  os << "activation_data_type: " << config.GetActivationDataType() << "\n";
  os << "max_num_images: " << config.GetMaxNumImages() << "\n";
  os << "cache_dir: " << config.GetCacheDir() << "\n";
  if (config.GetScopedCacheFile()) {
    os << "cache_file: " << config.GetScopedCacheFile()->file() << "\n";
  } else {
    os << "cache_file: Not set.\n";
  }
  os << "model_assets: " << config.GetModelAssets() << "\n";
  return os;
}

}  // namespace litert::lm
