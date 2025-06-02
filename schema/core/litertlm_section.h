#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_SECTION_H
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_SECTION_H

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert {
namespace lm {
namespace schema {

// Abstract base class for section streams
//
// A "section stream" represents a source of data that can be read sequentially.
// This data can be a file, a serialized protocol buffer, or any other
// contiguous block of bytes. The purpose of this abstraction is to allow
// different data sources to be handled uniformly by a reader or writer.
//
// Example Usage:
//
//   // Create a concrete stream (e.g., from a file)
//   std::unique_ptr<SectionStreamBase> stream =
//       std::make_unique<FileBackedSectionStream>("my_data.bin");
//
//   // Prepare the stream
//   absl::Status status = stream->Prepare();
//   if (!status.ok()) {
//     // Handle error
//   }
//
//   // Get the input stream
//   std::istream& istream = stream->GetStream();
//
//   // Read data from the stream
//   std::string data;
//   std::getline(istream, data);
//
//   // Finalize the stream
//   status = stream->Finalize();
//   if (!status.ok()) {
//     // Handle error
//   }
//
// This pattern allows a parser to work with different data sources without
// needing to know the specifics of how each source is handled.
class SectionStreamBase {
 public:
  // Virtual destructor
  virtual ~SectionStreamBase() = default;

  // Prepare: Pure virtual function to prepare the stream.
  virtual absl::Status Prepare() = 0;

  // GetStream: Pure virtual function to get the input stream.
  virtual std::istream& GetStream() = 0;

  // IsReady:  Check if the stream is ready for reading.
  virtual bool IsReady() const = 0;

  // Finalize:  Pure virtual function to finalize the stream.
  virtual absl::Status Finalize() = 0;

  // BufferSize:  Pure virtual function get the size of the streamed buffer.
  virtual size_t BufferSize() const = 0;
};

// A basic derived class for a file-backed stream. Reads the provided file
// during the Prepare(), holds it in an internal buffer, and then provides
// an input string stream for the caller to stream its contents.
class FileBackedSectionStream : public SectionStreamBase {
 public:
  // Constructor: Takes the file path.
  explicit FileBackedSectionStream(const std::string& file_path)
      : file_path_(file_path), buffer_(nullptr), buffer_size_(0) {}

  ~FileBackedSectionStream() override = default;

  // Prepare: Reads the file and prepares the internal buffer.  This function
  // *must* be called before using the stream.
  absl::Status Prepare() override {
    if (buffer_) {
      ABSL_LOG(INFO) << "Buffer already prepared for file: " << file_path_;
      return absl::OkStatus();
    }

    // Clear the stringstream before use.
    stream_.str(std::string());
    stream_.clear();

    std::ifstream file(file_path_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open file: ", file_path_));
    }

    buffer_size_ = static_cast<size_t>(file.tellg());  // Use size_t
    file.seekg(0, std::ios::beg);
    ABSL_VLOG(4) << "File size: " << buffer_size_ << " bytes.";

    buffer_.reset(new unsigned char[buffer_size_]);
    if (!file.read(reinterpret_cast<char*>(buffer_.get()), buffer_size_)) {
      buffer_.reset();  // Clean up on error. Set to null
      return absl::InternalError(
          absl::StrCat("Failed to read all data from file: ", file_path_));
    }
    ABSL_VLOG(4) << "Successfully read " << buffer_size_
                   << " bytes from file.";

    file.close();
    is_ready_ = true;
    stream_.write(reinterpret_cast<char*>(buffer_.get()), buffer_size_);
    stream_.seekg(0);
    ABSL_VLOG(4) << "Internal stringstream prepared.";
    return absl::OkStatus();
  }

  // Expose a stream-like object.
  std::istream& GetStream() override {
    if (!is_ready_) {
      ABSL_LOG(ERROR) << "Attempting to get stream before preparation.";
    }
    return stream_;
  }

  bool IsReady() const override { return is_ready_; }

  size_t BufferSize() const override { return buffer_size_; }

  absl::Status Finalize() override {
    if (buffer_) {
      buffer_.reset();  // Release the memory
      buffer_size_ = 0;
      is_ready_ = false;
      stream_.str(std::string());  // Clear the stringstream
      stream_.clear();             // Clear any error flags
      ABSL_LOG(INFO) << "Buffer finalized and stream reset for file: "
                     << file_path_;
    } else {
      ABSL_LOG(INFO) << "Nothing to finalize. Either Prepare() was not called "
                     << "or Finalize() has already been called.";
    }
    return absl::OkStatus();
  }

 private:
  std::string file_path_;
  std::unique_ptr<unsigned char[]> buffer_;
  size_t buffer_size_;
  bool is_ready_ = false;  // Track preparation state
  std::stringstream stream_;
};

// Class template for a stream backed by a protocol buffer.
// This class is particularly useful when a section of data is directly
// represented as a protocol buffer object in memory. Instead of writing this
// object to a file and then reading it back, this class allows to serialize
// the protocol buffer directly into a stream, which can then be used by the
// writer. This approach is more efficient as it avoids the overhead of file
// I/O and the need for temporary files.
template <typename T>
class ProtoBufSectionStream : public SectionStreamBase {
 public:
  // Constructor: Own a copy of the protocol buffer object, so that
  // this object can guarantee the ability to stream the contents
  // and release the memory upon destruction.
  explicit ProtoBufSectionStream(T proto)
      : proto_(std::move(proto)), is_ready_(false) {}

  ~ProtoBufSectionStream() override = default;

  // Prepare: Serializes the protocol buffer to a string.
  absl::Status Prepare() override {
    if (is_ready_) {
      ABSL_LOG(INFO) << "Stream already prepared for proto.";
      return absl::OkStatus();
    }

    // Clear the stringstream before use.
    stream_.str(std::string());
    stream_.clear();

    // Write directly into the stringstream's buffer.
    if (!proto_.SerializeToOstream(&stream_)) {
      return absl::InternalError("Failed to serialize protocol buffer.");
    }
    serialized_size_ =
        stream_.str()
            .size();  // Get the size from the stringstream's underlying string.
    is_ready_ = true;
    ABSL_LOG(INFO)
        << "Protocol buffer serialized directly to stringstream, size: "
        << serialized_size_ << " bytes.";
    return absl::OkStatus();
  }

  // GetStream: Returns a reference to the internal string stream.
  std::istream& GetStream() override {
    if (!is_ready_) {
      ABSL_LOG(ERROR) << "Attempting to get stream before preparation.";
    }
    return stream_;
  }

  bool IsReady() const override { return is_ready_; }

  absl::Status Finalize() override {
    stream_.str(std::string());
    stream_.clear();
    serialized_size_ = 0;
    is_ready_ = false;
    ABSL_LOG(INFO) << "Stream finalized.";
    return absl::OkStatus();
  }

  size_t BufferSize() const override { return serialized_size_; }

 private:
  T proto_;
  std::stringstream stream_;
  bool is_ready_;
  size_t serialized_size_;
};

}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_SECTION_H
