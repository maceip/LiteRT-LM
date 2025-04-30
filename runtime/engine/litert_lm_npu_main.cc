// A simple command line tool to run the litert LLM engine on NPU.

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_basic.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "tensorflow/lite/profiling/time.h"  // from @org_tensorflow

#ifndef __ANDROID__
// Float CPU models.
constexpr char kModelPathLlm[] = "gemma3_npu_f32_ekv1280.tflite";
constexpr char kModelPathTokenizer[] = "gemma3_tokenizer.spiece";
constexpr char kModelPathEmbedder[] = "gemma3_npu_embedder.tflite";
constexpr char kModelPathAuxiliary[] = "gemma3_npu_auxiliary.tflite";
#else
// Quantized CPU model (as in, not AOT compiled for NPU).
// constexpr char kModelPathLlm[] =
//     "static_a16w4-float-rms-gelu_quantized_gemma3_npu_f32_ekv1280.tflite";

// Quantized AOT compiled NPU model, optimized for sm8650.
constexpr char kModelPathLlm[] =
    "static_a16w4-full-int_quantized_gemma3_npu_f32_ekv1280_sm8750.tflite";

constexpr char kModelPathTokenizer[] = "gemma3_tokenizer.spiece";
constexpr char kModelPathEmbedder[] = "gemma3_npu_embedder.tflite";
constexpr char kModelPathAuxiliary[] = "gemma3_npu_auxiliary.tflite";
#endif

ABSL_FLAG(std::string, gemma3_path, "", "Path to the Gemma3 model.");
ABSL_FLAG(std::string, embedder_path, "", "Path to the embedder model.");
ABSL_FLAG(std::string, auxiliary_path, "", "Path to the auxiliary model.");
ABSL_FLAG(std::string, tokenizer_path, "", "Path to the tokenizer model.");
ABSL_FLAG(std::string, litert_dispatch_lib_path, "",
          "Path to the LiteRT dispatch library.");
ABSL_FLAG(std::string, prompt, "", "Prompt to run.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Create the tokenizer.
  auto tokenizer_or = litert::lm::SentencePieceTokenizer::CreateFromFile(
      absl::GetFlag(FLAGS_tokenizer_path));
  if (tokenizer_or.ok()) {
    ABSL_LOG(INFO) << "tokenizer created successfully";
  } else {
    ABSL_LOG(ERROR) << "tokenizer creation failed: " << tokenizer_or.status();
  }
  std::shared_ptr<litert::lm::Tokenizer> tokenizer =
      std::move(tokenizer_or.value());

  // Create the executor.
  uint64_t start = tflite::profiling::time::NowMicros();
  ABSL_LOG(INFO) << "Creating executor";
  auto executor_or = odml::infra::LlmLiteRtNpuCompiledModelExecutor::Create(
      absl::GetFlag(FLAGS_gemma3_path), absl::GetFlag(FLAGS_embedder_path),
      absl::GetFlag(FLAGS_auxiliary_path),
      absl::GetFlag(FLAGS_litert_dispatch_lib_path));
  uint64_t end = tflite::profiling::time::NowMicros();
  ABSL_LOG(INFO) << "executor creation took " << (end - start) << " us";
  if (executor_or.ok()) {
    ABSL_LOG(INFO) << "executor created successfully";
  } else {
    ABSL_LOG(ERROR) << "executor creation failed: " << executor_or.status();
  }
  std::unique_ptr<odml::infra::LlmLiteRtNpuCompiledModelExecutor> executor =
      std::move(executor_or.value());
  std::shared_ptr<odml::infra::LlmLiteRtNpuCompiledModelExecutor>
      executor_shared = std::move(executor);

  // Create the session.
  constexpr int kEndOfTurnTokenId = 106;
  std::vector<int> stop_token_ids = {kEndOfTurnTokenId};
  auto session = litert::lm::SessionBasic::Create(
      executor_shared, tokenizer, stop_token_ids,
      litert::lm::SessionConfig::CreateDefault());

  // Run the session.
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  ABSL_LOG(INFO) << "Prompt: " << prompt;
  start = tflite::profiling::time::NowMicros();
  auto status = (*session)->RunPrefill(prompt);
  end = tflite::profiling::time::NowMicros();
  ABSL_LOG(INFO) << "RunPrefill took " << (end - start) << " us";

  start = tflite::profiling::time::NowMicros();
  auto responses = (*session)->RunDecode();
  end = tflite::profiling::time::NowMicros();

  ABSL_LOG(INFO) << "RunDecode took " << (end - start) << " us";
  if (responses.ok()) {
    for (int i = 0; i < responses->GetNumOutputCandidates(); ++i) {
      auto response_text = responses->GetResponseTextAt(i);
      ABSL_LOG(INFO) << "Generated response: " << (*response_text);
    }
  } else {
    ABSL_LOG(ERROR) << "response failed: " << responses.status();
  }

  executor_shared->PrintLatencyStats();

  return 0;
}
