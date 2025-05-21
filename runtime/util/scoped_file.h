// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SCOPED_FILE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SCOPED_FILE_H_

#if defined(_WIN32)
#include <Windows.h>
#endif

#include <cstddef>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// A file wrapper that will automatically close on deletion.
class ScopedFile {
 public:
#if defined(_WIN32)
  using PlatformFile = HANDLE;
  static const PlatformFile kInvalidPlatformFile;
#else
  using PlatformFile = int;
  static constexpr PlatformFile kInvalidPlatformFile = -1;
#endif

  static absl::StatusOr<ScopedFile> Open(absl::string_view path);
  static absl::StatusOr<ScopedFile> OpenWritable(absl::string_view path);

  ScopedFile() : file_(kInvalidPlatformFile) {}
  explicit ScopedFile(PlatformFile file) : file_(file) {}
  ~ScopedFile() {
    if (IsValid()) {
      CloseFile(file_);
    }
  }

  ScopedFile(ScopedFile&& other) { file_ = other.Release(); }
  ScopedFile& operator=(ScopedFile&& other) {
    file_ = other.Release();
    return *this;
  }

  ScopedFile(const ScopedFile&) = delete;
  ScopedFile& operator=(const ScopedFile&) = delete;

  PlatformFile file() const { return file_; }
  bool IsValid() const { return file_ != kInvalidPlatformFile; }

  // Returns the number of bytes of the file.
  static absl::StatusOr<size_t> GetSize(PlatformFile file);
  absl::StatusOr<size_t> GetSize() const { return GetSize(file_); }

 private:
  PlatformFile Release() {
    PlatformFile temp = file_;
    file_ = kInvalidPlatformFile;
    return temp;
  }

  // Platform-specific file operations requiring platform-specific
  // implementations. It may be assumed by the implementation that the passed
  // `PlatformFile` is valid. This must be ensured by the caller.
  static void CloseFile(PlatformFile file);
  static absl::StatusOr<size_t> GetSizeImpl(PlatformFile file);

  PlatformFile file_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SCOPED_FILE_H_
