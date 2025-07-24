# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core library with shared constants and utilities for LiteRT-LM tools."""

import os
from litert_lm.schema.core import litertlm_header_schema_py_generated as schema

# --- File Format Constants ---
LITERTLM_MAJOR_VERSION = 1
LITERTLM_MINOR_VERSION = 3
LITERTLM_PATCH_VERSION = 0
BLOCK_SIZE = 16 * 1024
HEADER_BEGIN_BYTE_OFFSET = 32
HEADER_END_LOCATION_BYTE_OFFSET = 24

# --- Section Name Constants ---
K_TOKENIZER_SECTION_NAME = "tokenizer"
K_TFLITE_SECTION_NAME = "tflite"
K_LLM_METADATA_SECTION_NAME = "llm_metadata"
K_BINARY_DATA_SECTION_NAME = "binary_data"
K_HF_TOKENIZER_ZLIB_SECTION_NAME = "hf_tokenizer_zlib"


def any_section_data_type_to_string(data_type):
  """Converts AnySectionDataType enum to its string representation."""
  # Create a reverse mapping from the enum's integer values to names
  if not hasattr(any_section_data_type_to_string, "map"):
    any_section_data_type_to_string.map = {
        v: k for k, v in schema.AnySectionDataType.__dict__.items()
    }
  return any_section_data_type_to_string.map.get(
      data_type, f"Unknown AnySectionDataType value ({data_type})"
  )


def get_file_extension(filename):
  """Returns the file extension from a filename."""
  return os.path.splitext(filename)[1]


def get_section_type_and_name(filename):
  """Determines the section type and name from the filename."""
  ext = get_file_extension(filename)
  if ext == ".tflite":
    return schema.AnySectionDataType.TFLiteModel, K_TFLITE_SECTION_NAME
  elif ext in (".pb", ".proto", ".pbtext", ".prototext"):
    return (
        schema.AnySectionDataType.LlmMetadataProto,
        K_LLM_METADATA_SECTION_NAME,
    )
  elif ext == ".spiece":
    return schema.AnySectionDataType.SP_Tokenizer, K_TOKENIZER_SECTION_NAME
  elif filename.endswith("tokenizer.json"):
    return (
        schema.AnySectionDataType.HF_Tokenizer_Zlib,
        K_HF_TOKENIZER_ZLIB_SECTION_NAME,
    )
  else:
    return (
        schema.AnySectionDataType.GenericBinaryData,
        K_BINARY_DATA_SECTION_NAME,
    )
