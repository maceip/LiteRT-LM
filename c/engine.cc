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

#include "c/engine.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"

namespace {
// An implementation of InferenceObservable that forwards calls to a C-style
// callback function. It manages its own lifetime, deleting itself when the
// stream is complete.
class CCallbackObserver : public litert::lm::InferenceObservable {
 public:
  CCallbackObserver(LiteRtLmStreamCallback callback, void* callback_data)
      : callback_(callback), callback_data_(callback_data) {}

  // Called when a new response is generated.
  void OnNext(const litert::lm::Responses& responses) override {
    for (int i = 0; i < responses.GetNumOutputCandidates(); ++i) {
      auto response_text = responses.GetResponseTextAt(i);
      if (!response_text.ok()) {
        continue;
      }
      callback_(callback_data_, response_text->data(), false, nullptr);
    }
  }

  // Called when the inference is done and finished successfully.
  void OnDone() override {
    callback_(callback_data_, nullptr, true, nullptr);
    delete this;
  };

  void OnError(const absl::Status& status) override {
    if (callback_) {
      std::string error_message = status.ToString();
      callback_(callback_data_, nullptr, true, error_message.c_str());
    }
    delete this;
  }

 private:
  LiteRtLmStreamCallback callback_;
  void* callback_data_;
};
}  // namespace

using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputText;
using ::litert::lm::ModelAssets;
using ::litert::lm::Responses;
using ::litert::lm::SessionConfig;

struct LiteRtLmEngineSettings {
  std::unique_ptr<EngineSettings> settings;
};

struct LiteRtLmEngine {
  std::unique_ptr<Engine> engine;
};

struct LiteRtLmSession {
  std::unique_ptr<Engine::Session> session;
};

struct LiteRtLmResponses {
  Responses responses;
};

struct LiteRtLmBenchmarkInfo {
  litert::lm::BenchmarkInfo benchmark_info;
};

extern "C" {

LiteRtLmEngineSettings* litert_lm_engine_settings_create(
    const char* model_path, const char* backend_str) {
  auto model_assets = ModelAssets::Create(model_path);
  if (!model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create model assets: "
                    << model_assets.status();
    return nullptr;
  }
  auto backend = litert::lm::GetBackendFromString(backend_str);
  if (!backend.ok()) {
    ABSL_LOG(ERROR) << "Failed to parse backend: " << backend.status();
    return nullptr;
  }
  auto engine_settings =
      EngineSettings::CreateDefault(*std::move(model_assets), *backend);
  if (!engine_settings.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine settings: "
                    << engine_settings.status();
    return nullptr;
  }

  auto* c_settings = new LiteRtLmEngineSettings;
  c_settings->settings =
      std::make_unique<EngineSettings>(*std::move(engine_settings));
  return c_settings;
}

void litert_lm_engine_settings_delete(LiteRtLmEngineSettings* settings) {
  delete settings;
}

void litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings* settings, int max_num_tokens) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
        max_num_tokens);
  }
}

void litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings) {
  if (settings && settings->settings) {
    settings->settings->GetMutableBenchmarkParams();
  }
}

LiteRtLmEngine* litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings) {
  if (!settings || !settings->settings) {
    return nullptr;
  }

  auto engine = Engine::CreateEngine(*settings->settings);
  if (!engine.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine: " << engine.status();
    return nullptr;
  }

  auto* c_engine = new LiteRtLmEngine;
  c_engine->engine = *std::move(engine);
  return c_engine;
}
void litert_lm_engine_delete(LiteRtLmEngine* engine) { delete engine; }

LiteRtLmSession* litert_lm_engine_create_session(LiteRtLmEngine* engine) {
  if (!engine || !engine->engine) {
    return nullptr;
  }
  auto session = engine->engine->CreateSession(SessionConfig::CreateDefault());
  if (!session.ok()) {
    ABSL_LOG(ERROR) << "Failed to create session: " << session.status();
    return nullptr;
  }

  auto* c_session = new LiteRtLmSession;
  c_session->session = *std::move(session);
  return c_session;
}

void litert_lm_session_delete(LiteRtLmSession* session) { delete session; }

LiteRtLmResponses* litert_lm_session_generate_content(LiteRtLmSession* session,
                                                      const InputData* inputs,
                                                      size_t num_inputs) {
  if (!session || !session->session) {
    return nullptr;
  }
  std::vector<std::variant<litert::lm::InputText, litert::lm::InputImage,
                           litert::lm::InputAudio>>
      engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case kInputText:
        engine_inputs.emplace_back(InputText(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputImage:
        engine_inputs.emplace_back(litert::lm::InputImage(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputAudio:
        engine_inputs.emplace_back(litert::lm::InputAudio(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
    }
  }
  auto responses = session->session->GenerateContent(std::move(engine_inputs));
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to generate content: " << responses.status();
    return nullptr;
  }

  auto* c_responses = new LiteRtLmResponses{std::move(*responses)};
  return c_responses;
}

int litert_lm_session_generate_content_stream(LiteRtLmSession* session,
                                              const InputData* inputs,
                                              size_t num_inputs,
                                              LiteRtLmStreamCallback callback,
                                              void* callback_data) {
  if (!session || !session->session) {
    return -1;
  }
  std::vector<std::variant<litert::lm::InputText, litert::lm::InputImage,
                           litert::lm::InputAudio>>
      engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case kInputText:
        engine_inputs.emplace_back(InputText(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputImage:
        engine_inputs.emplace_back(litert::lm::InputImage(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputAudio:
        engine_inputs.emplace_back(litert::lm::InputAudio(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
    }
  }

  // The observer is heap-allocated and will manage its own lifetime.
  // It will be deleted in OnResponse(is_final=true) or OnError.
  auto* observer = new CCallbackObserver(callback, callback_data);

  absl::Status status = session->session->GenerateContentStream(
      std::move(engine_inputs), observer);

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start content stream: " << status;
    delete observer;  // Observer will not be called, so delete it here.
    return static_cast<int>(status.code());
  }
  return 0;  // The call is non-blocking and returns immediately.
}

void litert_lm_responses_delete(LiteRtLmResponses* responses) {
  delete responses;
}

int litert_lm_responses_get_num_candidates(const LiteRtLmResponses* responses) {
  if (!responses) {
    return 0;
  }
  return responses->responses.GetNumOutputCandidates();
}

const char* litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index) {
  if (!responses) {
    return nullptr;
  }
  auto response_text = responses->responses.GetResponseTextAt(index);
  if (!response_text.ok()) {
    return nullptr;
  }
  // The string_view's data is valid as long as the responses object is alive.
  return response_text->data();
}

LiteRtLmBenchmarkInfo* litert_lm_session_get_benchmark_info(
    LiteRtLmSession* session) {
  if (!session || !session->session) {
    return nullptr;
  }
  auto benchmark_info = session->session->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ABSL_LOG(ERROR) << "Failed to get benchmark info: "
                    << benchmark_info.status();
    return nullptr;
  }
  return new LiteRtLmBenchmarkInfo{std::move(*benchmark_info)};
}

void litert_lm_benchmark_info_delete(LiteRtLmBenchmarkInfo* benchmark_info) {
  delete benchmark_info;
}

double litert_lm_benchmark_info_get_time_to_first_token(
    const LiteRtLmBenchmarkInfo* benchmark_info) {
  if (!benchmark_info) {
    return 0.0;
  }
  return benchmark_info->benchmark_info.GetTimeToFirstToken();
}

int litert_lm_benchmark_info_get_num_prefill_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info) {
  if (!benchmark_info) {
    return 0;
  }
  return benchmark_info->benchmark_info.GetTotalPrefillTurns();
}

int litert_lm_benchmark_info_get_num_decode_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info) {
  if (!benchmark_info) {
    return 0;
  }
  return benchmark_info->benchmark_info.GetTotalDecodeTurns();
}

double litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index) {
  if (!benchmark_info) {
    return 0.0;
  }
  return benchmark_info->benchmark_info.GetPrefillTokensPerSec(index);
}

double litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index) {
  if (!benchmark_info) {
    return 0.0;
  }
  return benchmark_info->benchmark_info.GetDecodeTokensPerSec(index);
}

}  // extern "C"
