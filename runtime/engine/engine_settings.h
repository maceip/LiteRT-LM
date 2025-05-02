#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_

#include <optional>

#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Settings used for initializing LiteRT LM.
// This class encapsulates the model-specific settings that are used for
// initializing the LiteRT LM. These settings are typically fixed for a given
// model and are not expected to change during the inference process.
// The model assets are required to initialize the LiteRT LM.
// TODO(b/397975034) Add overloading << operator for debugging.
class EngineSettings {
 public:
  explicit EngineSettings(
      const LlmExecutorSettings& executor_settings,
      std::optional<proto::BenchmarkParams> benchmark_params = std::nullopt);

  const LlmExecutorSettings& GetMainExecutorSettings() const;

  // Benchmark parameters:
  // Returns true if the benchmark is enabled.
  bool IsBenchmarkEnabled() const;
  // Returns the benchmark parameters.
  std::optional<proto::BenchmarkParams> GetBenchmarkParams() const;
  // Sets the benchmark parameters.
  void SetBenchmarkParams(const proto::BenchmarkParams& benchmark_params);

 private:
  // Settings for the main executor.
  LlmExecutorSettings main_executor_settings_;

  // Parameters used to configure the benchmarking process.
  std::optional<proto::BenchmarkParams> benchmark_params_;
};

// Configurations used for the session.
// This class encapsulates the session-specific configurations that are used for
// creating a LiteRT LM session.
class SessionConfig {
 public:
  // Creates a default SessionConfig.
  static SessionConfig CreateDefault();

  // Creates a SessionConfig with the given sampler parameters.
  explicit SessionConfig(const proto::SamplerParameters& sampler_params);

  // Sampler parameters:
  // Returns the sampler parameters.
  proto::SamplerParameters GetSamplerParams() const;
  // Sets the sampler parameters.
  void SetSamplerParams(const proto::SamplerParameters& sampler_params);

 private:
  // Parameters used to configure the sampling process.
  proto::SamplerParameters sampler_params_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
