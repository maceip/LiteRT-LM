#include "runtime/engine/engine_settings.h"

#include <optional>

#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

EngineSettings::EngineSettings(
    const LlmExecutorSettings& executor_settings,
    std::optional<proto::BenchmarkParams> benchmark_params)
    : main_executor_settings_(executor_settings),
      benchmark_params_(benchmark_params) {}

const LlmExecutorSettings& EngineSettings::GetMainExecutorSettings() const {
  return main_executor_settings_;
}

// Benchmark parameters:
// Returns true if the benchmark is enabled.
bool EngineSettings::IsBenchmarkEnabled() const {
  return benchmark_params_.has_value();
}
// Returns the benchmark parameters.
std::optional<proto::BenchmarkParams> EngineSettings::GetBenchmarkParams()
    const {
  return benchmark_params_;
}
// Sets the benchmark parameters.
void EngineSettings::SetBenchmarkParams(
    const proto::BenchmarkParams& benchmark_params) {
  benchmark_params_ = benchmark_params;
}

SessionConfig SessionConfig::CreateDefault() {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.95f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  return SessionConfig(sampler_params);
}

SessionConfig::SessionConfig(const proto::SamplerParameters& sampler_params)
    : sampler_params_(sampler_params) {}

// Returns the sampler parameters.
proto::SamplerParameters SessionConfig::GetSamplerParams() const {
  return sampler_params_;
}
void SessionConfig::SetSamplerParams(
    const proto::SamplerParameters& sampler_params) {
  sampler_params_ = sampler_params;
}

}  // namespace litert::lm
