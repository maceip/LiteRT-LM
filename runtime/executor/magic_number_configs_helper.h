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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MAGIC_NUMBER_CONFIGS_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MAGIC_NUMBER_CONFIGS_HELPER_H_

#include <cstdlib>
#include <memory>
#include <vector>

#include "litert/c/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Helper to build magic number configs for LiteRT environment options when the
// model contains magic numbers for context length and prefill length.
class MagicNumberConfigsHelper {
 public:
  MagicNumberConfigsHelper(const Model& litert_model,
                           const LlmExecutorSettings& executor_settings);
  ~MagicNumberConfigsHelper() = default;

  // Returns LiteRT environment options including magic number configs and
  // verifications if any.
  std::vector<Environment::Option> GetLiteRtEnvOptions() const;

  const LiteRtMagicNumberConfigs* magic_number_configs() const {
    return magic_number_configs_.get();
  }
  const LiteRtMagicNumberVerifications* magic_number_verifications() const {
    return magic_number_verifications_.get();
  }

 private:
  struct FreeDeleter {
    void operator()(void* ptr) const { free(ptr); }
  };

  template <typename T>
  using UniqueCPtr = std::unique_ptr<T, FreeDeleter>;

  UniqueCPtr<LiteRtMagicNumberConfigs> magic_number_configs_;
  UniqueCPtr<LiteRtMagicNumberVerifications> magic_number_verifications_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MAGIC_NUMBER_CONFIGS_HELPER_H_
