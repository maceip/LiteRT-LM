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

#include "runtime/util/litert_lm_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

namespace {
// Utility function to Creates a memory-mapped file from a ScopedFile.
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> CreateMemoryMapFromScopedFile(
    litert::lm::ScopedFile& scoped_file) {
  if (!scoped_file.IsValid()) {
    return absl::InvalidArgumentError("Invalid ScopedFile provided.");
  }
  litert::lm::ScopedFile::PlatformFile platform_file = scoped_file.file();
  // For a read-only memory-mapped file:
  return litert::lm::MemoryMappedFile::Create(platform_file, 0, 0, "whole");
}

constexpr uint64_t kLitertLmHeaderMaxSize = 16 * 1024;

}  // namespace

absl::Status LitertLmLoader::MapSections() {
  litertlm::schema::LitertlmHeader header;
  int major_version, minor_version, patch_version;

  // Read the header information.
  absl::Status status = ReadHeaderFromLiteRTLM(
      memory_mapped_file_->data(),
      std::min(kLitertLmHeaderMaxSize, memory_mapped_file_->length()), &header,
      &major_version, &minor_version, &patch_version);
  ABSL_LOG(INFO) << "status: " << status;
  ABSL_LOG(INFO) << "major_version: " << major_version;
  ABSL_LOG(INFO) << "minor_version: " << minor_version;
  ABSL_LOG(INFO) << "patch_version: " << patch_version;

  if (!status.ok()) {
    return status;
  }

  // Loop through the sections and map them to the section buffers.
  auto sections = header.metadata->section_metadata()->objects();
  for (size_t i = 0; i < sections->size(); ++i) {
    const litertlm::schema::SectionObject* section = sections->Get(i);
    section_buffers_[section->data_type()] =
        BufferRef<uint8_t>(static_cast<uint8_t*>(memory_mapped_file_->data()),
                           section->end_offset(), section->begin_offset());
    ABSL_LOG(INFO) << "section_index: " << i;
    ABSL_LOG(INFO) << "section_data_type: "
                   << EnumNameAnySectionDataType(section->data_type());
    ABSL_LOG(INFO) << "section_begin_offset: " << section->begin_offset();
    ABSL_LOG(INFO) << "section_end_offset: " << section->end_offset();
  }
  return absl::OkStatus();
}

absl::Status LitertLmLoader::Initialize() {
  ABSL_LOG(INFO) << "LitertLmLoader::Initialize";

  absl::StatusOr<std::unique_ptr<MemoryMappedFile>> mmap_status =
      CreateMemoryMapFromScopedFile(model_file_);

  if (mmap_status.ok()) {
    memory_mapped_file_ = std::move(mmap_status).value();
    ABSL_LOG(INFO) << "mmap_status is ok";
    ABSL_LOG(INFO) << "length: " << memory_mapped_file_->length();
  } else {
    ABSL_LOG(ERROR) << "Failed to create memory-mapped file: "
                    << mmap_status.status();
  }

  ABSL_CHECK_OK(MapSections());

  return absl::OkStatus();
}

}  // namespace litert::lm
