#include "runtime/engine/io_types.h"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {

// A container to host the model responses.
Responses::Responses(int num_output_candidates)
    : num_output_candidates_(num_output_candidates) {
  response_texts_ = std::vector<std::string>(num_output_candidates_);
}

absl::StatusOr<absl::string_view> Responses::GetResponseTextAt(
    int index) const {
  if (index < 0 || index >= num_output_candidates_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Index ", index, " is out of range [0, ",
                     num_output_candidates_, ")."));
  }
  return response_texts_[index];
}

absl::StatusOr<float> Responses::GetScoreAt(int index) const {
  if (scores_.empty()) {
    return absl::InvalidArgumentError("Scores are not set.");
  }
  if (index < 0 || index >= scores_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Index ", index, " is out of range [0, ", scores_.size(), ")."));
  }
  return scores_[index];
}

std::vector<std::string>& Responses::GetMutableResponseTexts() {
  return response_texts_;
}

std::vector<float>& Responses::GetMutableScores() {
  if (scores_.empty()) {
    scores_ = std::vector<float>(num_output_candidates_,
                                 -std::numeric_limits<float>::infinity());
  }
  return scores_;
}

std::ostream& operator<<(std::ostream& os, const Responses& responses) {
  if (responses.GetNumOutputCandidates() == 0) {
    os << " No reponses." << std::endl;
    return os;
  }
  os << "Total candidates: " << responses.GetNumOutputCandidates() << ":"
     << std::endl;

  for (int i = 0; i < responses.GetNumOutputCandidates(); ++i) {
    absl::StatusOr<float> score_status = responses.GetScoreAt(i);
    if (score_status.ok()) {
      os << "  Candidate " << i << " (score: " << *score_status
         << "):" << std::endl;
    } else {
      os << "  Candidate " << i << " (score: N/A):" << std::endl;
    }

    absl::StatusOr<absl::string_view> text_status =
        responses.GetResponseTextAt(i);
    if (text_status.ok()) {
      os << "    Text: \"" << *text_status << "\"" << std::endl;
    } else {
      os << "    Text: Error - " << text_status.status().message() << std::endl;
    }
  }
  return os;  // Return the ostream to allow chaining
}

// --- BenchmarkTurnData Method Definitions ---
BenchmarkTurnData::BenchmarkTurnData(uint64_t tokens, absl::Duration dur)
    : duration(dur), num_tokens(tokens) {}

BenchmarkInfo::BenchmarkInfo(const proto::BenchmarkParams& benchmark_params)
    : benchmark_params_(benchmark_params) {};

const proto::BenchmarkParams& BenchmarkInfo::GetBenchmarkParams() const {
  return benchmark_params_;
}

absl::Status BenchmarkInfo::TimeInitPhaseStart(const std::string& phase_name) {
  if (start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Phase ", phase_name, " already started."));
  }
  start_time_map_[phase_name] = absl::Now();
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimeInitPhaseEnd(const std::string& phase_name) {
  if (!start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Phase ", phase_name, " not started."));
  }
  init_phases_[phase_name] = absl::Now() - start_time_map_[phase_name];
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimePrefillTurnStart() {
  const std::string phase_name = absl::StrCat("prefill:", prefill_turn_index_);
  if (start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Prefill turn ", phase_name, " already started."));
  }
  start_time_map_[phase_name] = absl::Now();
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimePrefillTurnEnd(uint64_t num_prefill_tokens) {
  const std::string phase_name = absl::StrCat("prefill:", prefill_turn_index_);
  if (!start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Prefill turn ", phase_name, " not started."));
  }
  prefill_turns_.emplace_back(num_prefill_tokens,
                              absl::Now() - start_time_map_[phase_name]);
  prefill_turn_index_++;
  return absl::OkStatus();
}

const BenchmarkTurnData& BenchmarkInfo::GetPrefillTurn(int turn_index) const {
  return prefill_turns_[turn_index];
}

absl::Status BenchmarkInfo::TimeDecodeTurnStart() {
  const std::string phase_name = absl::StrCat("decode:", decode_turn_index_);
  if (start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Decode turn ", phase_name, " already started."));
  }
  start_time_map_[phase_name] = absl::Now();
  return absl::OkStatus();
}

absl::Status BenchmarkInfo::TimeDecodeTurnEnd(uint64_t num_decode_tokens) {
  const std::string phase_name = absl::StrCat("decode:", decode_turn_index_);
  if (!start_time_map_.contains(phase_name)) {
    return absl::InternalError(
        absl::StrCat("Decode turn ", phase_name, " not started."));
  }
  decode_turns_.emplace_back(num_decode_tokens,
                             absl::Now() - start_time_map_[phase_name]);
  decode_turn_index_++;
  return absl::OkStatus();
}

const BenchmarkTurnData& BenchmarkInfo::GetDecodeTurn(int turn_index) const {
  return decode_turns_[turn_index];
}

const std::map<std::string, absl::Duration>& BenchmarkInfo::GetInitPhases()
    const {
  return init_phases_;
}

uint64_t BenchmarkInfo::GetTotalPrefillTurns() const {
  return prefill_turns_.size();
}

double BenchmarkInfo::GetPrefillTokensPerSec(int turn_index) const {
  if (turn_index < 0 ||
      static_cast<size_t>(turn_index) >= prefill_turns_.size()) {
    return 0.0;
  }
  const auto& turn = prefill_turns_[turn_index];
  if (turn.duration <= absl::ZeroDuration()) {
    return 0.0;  // Avoid division by zero or negative duration
  }
  double turn_seconds = absl::ToDoubleSeconds(turn.duration);
  if (turn_seconds <= 0.0) {  // Additional check for very small durations
    return 0.0;
  }
  return static_cast<double>(turn.num_tokens) / turn_seconds;
}

uint64_t BenchmarkInfo::GetTotalDecodeTurns() const {
  return decode_turns_.size();
}

// Interpreted as Generated Tokens Per Second for the specified turn_index.
// The "Avg" in the name might be a misnomer if it's for a specific turn,
// but implementing based on the header's declaration.
double BenchmarkInfo::GetDecodeTokensPerSec(int turn_index) const {
  if (turn_index < 0 ||
      static_cast<size_t>(turn_index) >= decode_turns_.size()) {
    // Consider logging an error or throwing std::out_of_range
    return 0.0;
  }
  const auto& turn = decode_turns_[turn_index];
  if (turn.duration <= absl::ZeroDuration()) {
    return 0.0;  // Avoid division by zero or negative duration
  }
  double turn_seconds = absl::ToDoubleSeconds(turn.duration);
  if (turn_seconds <= 0.0) {  // Additional check for very small durations
    return 0.0;
  }
  // This calculates tokens/sec for the specific turn.
  // If "turns/sec" for a specific turn was intended, the logic would be
  // different (1.0 / turn_seconds). Given the name and typical metrics,
  // tokens/sec for the turn seems more likely.
  return static_cast<double>(turn.num_tokens) / turn_seconds;
}

std::ostream& operator<<(std::ostream& os, const BenchmarkTurnData& data) {
  os << "Processed " << data.num_tokens << " tokens in " << data.duration
     << " duration." << std::endl;
  return os;
}

std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info) {
  os << std::fixed << std::setprecision(2);

  os << "BenchmarkInfo:" << std::endl;
  os << "  Init Phases (" << info.GetInitPhases().size() << "):" << std::endl;
  if (info.GetInitPhases().empty()) {
    os << "    No init phases recorded." << std::endl;
  } else {
    double total_time = 0.0;
    for (const auto& phase : info.GetInitPhases()) {
      total_time += absl::ToDoubleMilliseconds(phase.second);
      os << "    - " << phase.first << ": "
         << absl::ToDoubleMilliseconds(phase.second) << " ms" << std::endl;
    }
    os << "    Total init time: " << total_time << " ms" << std::endl;
  }

  os << "--------------------------------------------------" << std::endl;
  os << "  Prefill Turns (Total: " << info.GetTotalPrefillTurns()
     << "):" << std::endl;
  if (info.GetTotalPrefillTurns() == 0) {
    os << "    No prefill turns recorded." << std::endl;
  } else {
    for (uint64_t i = 0; i < info.GetTotalPrefillTurns(); ++i) {
      os << "    Prefill Turn " << i + 1 << ": " << info.GetPrefillTurn(i);
      os << "      Prefill Speed: "
         << info.GetPrefillTokensPerSec(static_cast<int>(i)) << " tokens/sec."
         << std::endl;
    }
  }

  os << "--------------------------------------------------" << std::endl;
  os << "  Decode Turns (Total: " << info.GetTotalDecodeTurns()
     << "):" << std::endl;
  if (info.GetTotalDecodeTurns() == 0) {
    os << "    No decode turns recorded." << std::endl;
  } else {
    for (uint64_t i = 0; i < info.GetTotalDecodeTurns(); ++i) {
      os << "    Decode Turn " << i + 1 << ": " << info.GetDecodeTurn(i);
      os << "      Decode Speed: "
         << info.GetDecodeTokensPerSec(static_cast<int>(i)) << " tokens/sec."
         << std::endl;
    }
  }
  os << "--------------------------------------------------" << std::endl;

  return os;
}

}  // namespace litert::lm
