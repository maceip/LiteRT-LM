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

#include "runtime/util/lora_data.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "tflite/model_builder.h"  // from @litert
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

// LoRA data based on FlatBufferModel.
class FlatBufferLoraData : public LoraData {
 public:
  ~FlatBufferLoraData() override = default;

 protected:
  // Returns the FlatBufferModel object.
  // FlatBufferModel is owned by derived classes to be destroyed in correct
  // order, thus it is accessed by base class with a reference here.
  virtual std::shared_ptr<tflite::FlatBufferModel> GetModel() const = 0;
};

// FlatBufferModel based LoRA data backed by a file.
class FileLoraData : public FlatBufferLoraData {
 public:
  // Constructor for FileLoraData.
  //
  // @param file A shared_ptr to the ScopedFile object representing the LoRA
  // data file.
  // @param region A unique_ptr to the MemoryMappedFile object representing the
  // memory mapped region of the file.
  // @param model A shared_ptr to the FlatBufferModel object representing the
  // LoRA data.
  explicit FileLoraData(std::shared_ptr<const ScopedFile> file,
                        std::unique_ptr<MemoryMappedFile> region,
                        std::shared_ptr<tflite::FlatBufferModel> model)
      : file_(std::move(file)),
        region_(std::move(region)),
        model_(std::move(model)) {}

  ~FileLoraData() override = default;

 private:
  std::shared_ptr<tflite::FlatBufferModel> GetModel() const override {
    return model_;
  }

 private:
  std::shared_ptr<const ScopedFile> file_;
  std::unique_ptr<MemoryMappedFile> region_;
  std::shared_ptr<tflite::FlatBufferModel> model_;
};

}  // namespace

// static
absl::StatusOr<std::unique_ptr<LoraData>> LoraData::CreateFromFilePath(
    absl::string_view file_path) {
  ASSIGN_OR_RETURN(auto file, ScopedFile::Open(file_path));
  return CreateFromScopedFile(std::make_shared<ScopedFile>(std::move(file)));
}

// static
absl::StatusOr<std::unique_ptr<LoraData>> LoraData::CreateFromScopedFile(
    std::shared_ptr<const ScopedFile> file) {
  ASSIGN_OR_RETURN(auto mapped_file,
                   ::litert::lm::MemoryMappedFile::Create(file->file()));
  bool obfuscated = !tflite::ModelBufferHasIdentifier(mapped_file->data());
  if (obfuscated) {
    return absl::UnimplementedError(
        "Input is not valid flatbuffer model. Deobfuscation is not supported "
        "yet.");
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
          reinterpret_cast<const char*>(mapped_file->data()),
          mapped_file->length());
  RET_CHECK(model) << "Error building tflite model.";
  return std::make_unique<FileLoraData>(std::move(file), std::move(mapped_file),
                                        std::move(model));
}

}  // namespace litert::lm
