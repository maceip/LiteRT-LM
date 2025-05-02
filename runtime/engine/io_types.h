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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/proto/engine.pb.h"

namespace litert::lm {

// A container to host the model responses.
class Responses {
 public:
  explicit Responses(int num_output_candidates);

  // Returns the number of output candidates.
  int GetNumOutputCandidates() const { return num_output_candidates_; }

  // Returns the response text at the given index. Returns error if the index is
  // out of range.
  absl::StatusOr<absl::string_view> GetResponseTextAt(int index) const;

  // Returns the score at the given index. Returns error if the index is out of
  // range or if scores are not included.
  // Note that the "score" is calculated as the sum of the log probabilities of
  // the whole decoded sequence normalized by the total number of tokens.
  absl::StatusOr<float> GetScoreAt(int index) const;

  // Returns the mutable response texts vector.
  std::vector<std::string>& GetMutableResponseTexts();

  // Returns the mutable scores vector. If it is the first time calling this
  // function, the scores vector will be allocated to the size of
  // num_output_candidates_ and initialized to the default value of -Inf
  // (= log(0.0f)).
  std::vector<float>& GetMutableScores();

 private:
  // The number of output candidates.
  const int num_output_candidates_;

  // The output vector of response tokens (as strings).
  std::vector<std::string> response_texts_;

  // The output vector of scores for each response text. The "score" is pulled
  // from the probability of the last token in the response text.
  std::vector<float> scores_;
};
std::ostream& operator<<(std::ostream& os, const Responses& responses);

// Class to store the data for a single turn of the benchmark. A "turn" is
// defined as a single RunPrefill or RunDecode call.
struct BenchmarkTurnData {
  absl::Duration duration;  // Duration of this entire operation/turn.
  uint64_t num_tokens;      // The number of tokens processed in this turn.
  BenchmarkTurnData(uint64_t tokens, absl::Duration dur);
};
std::ostream& operator<<(std::ostream& os, const BenchmarkTurnData& data);

// Class to store and manage comprehensive performance benchmark information for
// LLMs.
class BenchmarkInfo {
 public:
  explicit BenchmarkInfo(const proto::BenchmarkParams& benchmark_params);
  const proto::BenchmarkParams& GetBenchmarkParams() const;

  // --- Methods to record data ---
  // Time the start and end of a phase in the initialization. The phase name
  // should be a string that uniquely identifies the phase. Otherwise, the
  // methods will return an error.
  absl::Status TimeInitPhaseStart(const std::string& phase_name);
  absl::Status TimeInitPhaseEnd(const std::string& phase_name);
  // Time the start and end of a prefill/decode turn. The num_prefill_tokens
  // should be the number of tokens processed in this turn. The method will
  // return an error if the methods are called out of order (i.e. one end after
  // one start).
  absl::Status TimePrefillTurnStart();
  absl::Status TimePrefillTurnEnd(uint64_t num_prefill_tokens);
  absl::Status TimeDecodeTurnStart();
  absl::Status TimeDecodeTurnEnd(uint64_t num_decode_tokens);

  // --- Getters for raw data ---
  const std::map<std::string, absl::Duration>& GetInitPhases() const;

  // --- Calculated metrics and getters for Prefill ---
  uint64_t GetTotalPrefillTurns() const;
  const BenchmarkTurnData& GetPrefillTurn(int turn_index) const;
  double GetPrefillTokensPerSec(int turn_index) const;

  // --- Calculated metrics and getters for Decode ---
  uint64_t GetTotalDecodeTurns() const;
  const BenchmarkTurnData& GetDecodeTurn(int turn_index) const;
  double GetDecodeTokensPerSec(int turn_index) const;

 private:
  proto::BenchmarkParams benchmark_params_;

  // Map of phase names to start time.
  std::map<std::string, absl::Time> start_time_map_;
  // The current index of the prefill / decode turn.
  int prefill_turn_index_ = 0;
  int decode_turn_index_ = 0;

  std::map<std::string, absl::Duration> init_phases_;
  std::vector<BenchmarkTurnData> prefill_turns_;
  std::vector<BenchmarkTurnData> decode_turns_;
};
std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
