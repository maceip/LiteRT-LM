#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_

#include <memory>
#include <optional>
#include <ostream>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Note for development conventions:
// 1. Any optional field should use std::optional.
// 2. All member variables should be private and have their corresponding
// getters and setters.
// 3. For basic types, e.g. int, float, bool, etc., the getters and setters
// should be Get*() and Set*().
// 4. For complex types, e.g. proto::BenchmarkParams, the getters and setters
// should be Get*() and GetMutable*().
// 5. For optional fields, the mutable getter should create a default instance
// if the field is not set. But the non-mutable getter should return a
// const reference to the std::optional<T> field.

// Settings used for initializing LiteRT LM Engine.
// This class encapsulates the model-specific settings that are used for
// initializing the LiteRT LM. These settings are typically fixed for a given
// model and are not expected to change during the inference process.
//
// This class is used to initialize the LiteRT LM Engine. The user should
// create an EngineSettings object and then call the MaybeUpdateAndValidate()
// method to validate the settings. If the validation fails, the user should
// not use the EngineSettings object.
//
// Example:
//
//   ASSIGN_OR_RETURN(ModelAssets model_assets,
//                    ModelAssets::Create(model_path));
//   ASSIGN_OR_RETURN(EngineSettings engine_settings,
//                    EngineSettings::CreateDefault(model_assets));
//    ...initialize the Engine...
//   ASSIGN_OR_RETURN(std::unique_ptr<Engine> engine,
//                    Engine::CreateEngine(engine_settings));
// TODO(b/397975034) Add overloading << operator for debugging.
class EngineSettings {
 public:
  // Creates a default EngineSettings with the given model assets and specified
  // backend.
  static absl::StatusOr<EngineSettings> CreateDefault(
      const ModelAssets& model_assets, Backend backend = Backend::CPU);

  // Updates the EngineSettings fields by loading the metadata from the model
  // assets. The function also validates to check if all of the required fields
  // are set correctly. Returns an error if the validation fails.
  absl::Status MaybeUpdateAndValidate(
      std::shared_ptr<Tokenizer> tokenizer,
      std::shared_ptr<proto::LlmMetadata> metadata_from_file);

  // Returns the LlmExecutorSettings.
  const LlmExecutorSettings& GetMainExecutorSettings() const;
  // Returns the mutable LlmExecutorSettings.
  LlmExecutorSettings& GetMutableMainExecutorSettings();

  // Benchmark parameters:
  // Returns true if the benchmark is enabled.
  bool IsBenchmarkEnabled() const;
  // Returns the benchmark parameters.
  const std::optional<proto::BenchmarkParams>& GetBenchmarkParams() const;
  // Returns the mutable benchmark parameters.
  proto::BenchmarkParams& GetMutableBenchmarkParams();

  // Returns the LlmMetadata parameters.
  const std::optional<proto::LlmMetadata>& GetLlmMetadata() const;
  // Returns the mutable LlmMetadata parameters. Note that is the metadata_ is
  // not set (i.e. std::nullopt), then the default LlmMetadata will be
  // created and returned.
  proto::LlmMetadata& GetMutableLlmMetadata();

 private:
  explicit EngineSettings(
      const LlmExecutorSettings& executor_settings,
      std::optional<proto::BenchmarkParams> benchmark_params = std::nullopt);

  // Settings for the main executor.
  LlmExecutorSettings main_executor_settings_;

  // Parameters used to configure the benchmarking process.
  std::optional<proto::BenchmarkParams> benchmark_params_;

  // Default metadata for the model. This is loaded from the model assets (if
  // present).
  std::optional<proto::LlmMetadata> metadata_;
};
std::ostream& operator<<(std::ostream& os, const EngineSettings& settings);

// Configurations used for the session.
// This class encapsulates the session-specific configurations that are used for
// creating a LiteRT LM session.
class SessionConfig {
 public:
  // Creates a default SessionConfig.
  static SessionConfig CreateDefault();

  // Updates the SessionConfig fields from the EngineSettings when not set. The
  // function also validates to check if all of the required fields are set
  // correctly. Returns an error if the validation fails.
  absl::Status MaybeUpdateAndValidate(const EngineSettings& engine_settings);

  // Sampler parameters:
  // Getters for the sampler parameters.
  const proto::SamplerParameters& GetSamplerParams() const;
  proto::SamplerParameters& GetMutableSamplerParams();

  // Stop token ids:
  // Getters for the stop token ids.
  const std::vector<std::vector<int>>& GetStopTokenIds() const;
  std::vector<std::vector<int>>& GetMutableStopTokenIds();

  // Set the start token ids.
  int GetStartTokenId() const;
  void SetStartTokenId(int start_token_id);

  // Number of output candidates:
  // Getters for the number of output candidates.
  int GetNumOutputCandidates() const;
  void SetNumOutputCandidates(int num_output_candidates);

 private:
  // Private constructor for the SessionConfig. The user should use the
  // CreateDefault() method to create a SessionConfig.
  explicit SessionConfig(const proto::SamplerParameters& sampler_params);

  // Parameters used to configure the sampling process.
  proto::SamplerParameters sampler_params_;

  // Stop token ids for the session. Note that the stop token could be a
  // sequence of token ids (as opposed to a single token id). The first
  // dimension is the index of the stop token in the session, and the second
  // dimension is the sequence of token ids that constitutes the stop token.
  std::vector<std::vector<int>> stop_token_ids_;

  // Start token id for the session.
  int start_token_id_;

  // The number of output candidates to generate. Default value is 1 and setting
  // it to a value greater than 1 will require the model to support batching.
  int num_output_candidates_;
};
std::ostream& operator<<(std::ostream& os, const SessionConfig& config);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
