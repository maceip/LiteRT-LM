#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert {

namespace lm {

namespace schema {

using litert::lm::proto::LlmMetadata;

struct LitertlmHeader {
  std::unique_ptr<uint8_t[]> buffer;
  const LiteRTLMMetaData* metadata;

  // Default constructor
  LitertlmHeader() : buffer(nullptr), metadata(nullptr) {}

  // Constructor that takes ownership of the buffer.
  explicit LitertlmHeader(std::unique_ptr<uint8_t[]>&& buffer_)
      : buffer(std::move(buffer_)) {
    reset(std::move(buffer_));
  }

  // Disable copy constructor and assignment operator to prevent incorrect
  // copying.
  LitertlmHeader(const LitertlmHeader&) = delete;
  LitertlmHeader& operator=(const LitertlmHeader&) = delete;

  // Add a move constructor and move assignment operator for proper transfer of
  // ownership
  LitertlmHeader(LitertlmHeader&& other) noexcept
      : buffer(std::move(other.buffer)), metadata(other.metadata) {
    other.metadata = nullptr;
  }

  // reset function
  void reset(std::unique_ptr<uint8_t[]>&& buffer_) {
    buffer = std::move(buffer_);
    if (buffer) {
      metadata = GetLiteRTLMMetaData(buffer.get());
    } else {
      metadata = nullptr;
    }
  }

  ~LitertlmHeader() {
    // No need to delete, unique_ptr handles it.
  }
};

// Reads the LiteRTLM file header starting at `data`. It is assumed
// that this function can read up to `length` bytes starting at `data`.
//
// Args:
//   data: The pointer to some buffer we can read from.
//   length: The number of bytes that it is valid to read from starting at
//           `data`
//   header: The LitertlmHeader struct with populated schema data.
//   major_version: The major version of the LiteRTLM file.
//   minor_version: The minor version of the LiteRTLM file.
//   patch_version: The patch version of the LiteRTLM file.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(void* data, size_t length,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version);

// Reads the LiteRTLM file from the given path and populates a header
// data structure (allocating and owning data for the header).
//
// Args:
//   litertlm_path: The path to the LiteRTLM file.
//   header: The LitertlmHeader struct with populated schema data.
//   major_version: The major version of the LiteRTLM file.
//   minor_version: The minor version of the LiteRTLM file.
//   patch_version: The patch version of the LiteRTLM file.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(const std::string& litertlm_path,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version);

// Reads the LiteRTLM file from the given istream and populates a header
// data structure (allocating and owning data for the header).
//
// Args:
//   litertlm_stream: The input stream to the LiteRTLM file.
//   header: The LitertlmHeader struct with populated schema data.
//   major_version: The major version of the LiteRTLM file.
//   minor_version: The minor version of the LiteRTLM file.
//   patch_version: The patch version of the LiteRTLM file.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version);
}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_
