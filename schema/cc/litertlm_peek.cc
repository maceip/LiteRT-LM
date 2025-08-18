// This tool is designed to inspect the contents of a LiteRT-LM file.
// It reads the file's header, system metadata, and section information,
// and prints them to the console.
//
// Example usage:
// bazel run :litertlm_peek -- --litertlm_file=/path/to/your/file.litertlm

#include <iostream>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "schema/core/litertlm_print.h"

ABSL_FLAG(std::string, litertlm_file, "",
          "The path to the LiteRT-LM file to inspect.");

namespace {

using litert::lm::schema::ProcessLiteRTLMFile;

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string litertlm_file = absl::GetFlag(FLAGS_litertlm_file);
  ABSL_LOG(INFO) << "LiteRT-LM file: " << litertlm_file << "\n";

  if (litertlm_file.empty()) {
    return absl::InvalidArgumentError("--litertlm_file must be provided.");
  }

  // Use std::cout as the output stream.
  return ProcessLiteRTLMFile(litertlm_file, std::cout);
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
