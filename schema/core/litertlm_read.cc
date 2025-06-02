#include "schema/core/litertlm_read.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_utils.h"
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert {
namespace lm {
namespace schema {

using litert::lm::proto::LlmMetadata;

absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version) {
  // 0. Read magic number and version.
  char magic_number[8];
  litertlm_stream.read(magic_number, 8);
  if (litertlm_stream.gcount() != 8 ||
      std::string(magic_number, 8) != "LITERTLM") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid magic number or failed to read: %s",
                        std::string(magic_number, litertlm_stream.gcount())));
  }

  uint8_t major_version_u8, minor_version_u8, patch_version_u8;
  litertlm_stream.read(reinterpret_cast<char*>(&major_version_u8),
                       sizeof(uint8_t));
  litertlm_stream.read(reinterpret_cast<char*>(&minor_version_u8),
                       sizeof(uint8_t));
  litertlm_stream.read(reinterpret_cast<char*>(&patch_version_u8),
                       sizeof(uint8_t));

  if (!litertlm_stream) {
    return absl::InternalError("Failed to read version bytes.");
  }

  // Store the version as ints.
  *major_version = static_cast<int>(major_version_u8);
  *minor_version = static_cast<int>(minor_version_u8);
  *patch_version = static_cast<int>(patch_version_u8);

  // If major version doesn't match our current major version,
  // bail out for now
  if (*major_version != LITERTLM_MAJOR_VERSION) {
    return absl::UnimplementedError(
        absl::StrFormat("Unimplemented Error: This reader doesn't support "
                        "version %d, expected version %d.",
                        *major_version, LITERTLM_MAJOR_VERSION));
  }

  // 1. Skip 5 bytes of padding.
  litertlm_stream.ignore(5);
  if (!litertlm_stream) {
    return absl::InternalError("Failed to skip padding after version.");
  }

  // 2. Read the header end offset.
  uint64_t header_end_offset;
  litertlm_stream.read(reinterpret_cast<char*>(&header_end_offset),
                       sizeof(uint64_t));
  if (!litertlm_stream) {
    return absl::InternalError("Failed to read header end offset.");
  }

  // 3. Skip 8 bytes of padding.
  litertlm_stream.ignore(8);
  if (!litertlm_stream) {
    return absl::InternalError(
        "Failed to skip padding after header end offset.");
  }

  // Calculate the header size.
  std::streampos current_position = litertlm_stream.tellg();
  if (current_position == -1) {
    return absl::InternalError("Failed to get current stream position.");
  }
  // Ensure header_end_offset is greater than or equal to current_position
  if (header_end_offset < static_cast<uint64_t>(current_position)) {
    return absl::InvalidArgumentError(
        "Invalid header end offset: smaller than current position.");
  }
  uint64_t header_size =
      header_end_offset - static_cast<uint64_t>(current_position);

  // 3. Read the header data into a buffer.
  auto header_buffer = std::make_unique<uint8_t[]>(header_size);

  litertlm_stream.read(reinterpret_cast<char*>(header_buffer.get()),
                       header_size);
  if (!litertlm_stream) {
    return absl::InternalError("Failed to read header data.");
  }

  header->reset(std::move(header_buffer));
  return absl::OkStatus();
}

// The public function that takes a file path.
absl::Status ReadHeaderFromLiteRTLM(const std::string& litertlm_path,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version) {
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }

  absl::Status status = ReadHeaderFromLiteRTLM(
      input_file_stream, header, major_version, minor_version, patch_version);

  return status;
}

// The public function that takes a pointer and a length.
absl::Status ReadHeaderFromLiteRTLM(void* data, std::size_t length,
                                    LitertlmHeader* header, int* major_version,
                                    int* minor_version, int* patch_version) {
  char* char_data = static_cast<char*>(data);
  // Create a streambuf instance based on the given buffer info.
  MemoryStreamBuf sbuf(char_data, length);
  // Create an istream using the custom streambuf.
  std::istream input_stream(&sbuf);

  absl::Status status = ReadHeaderFromLiteRTLM(
      input_stream, header, major_version, minor_version, patch_version);
  // Cleanup of the streambuf and istream is automatic upon exit.
  return status;
}

}  // namespace schema
}  // namespace lm
}  // namespace litert
