#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert {

namespace lm {

namespace schema {

// Send info about the LiteRT-LM file to the output stream.
absl::Status ProcessLiteRTLMFile(const std::string& litertlm_file,
                                 std::ostream& output_stream);

}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
