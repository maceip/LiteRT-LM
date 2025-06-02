#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_LITERTLM_WRITER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_LITERTLM_WRITER_UTILS_H_
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl

namespace litert::lm::schema {

absl::Status LitertLmWrite(const std::vector<std::string>& command_args,
                           const std::string& section_metadata_str,
                           const std::string& output_path);

}  // namespace litert::lm::schema
#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_LITERTLM_WRITER_UTILS_HU
