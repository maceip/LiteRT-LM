# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for creating a LiteRT-LM file from a set of input files.

This library provides the core functionality for the `litertlm_writer` tool.
It handles parsing of input files and metadata, constructs the LiteRT-LM file
structure with a header and data sections, and writes the final binary file
to disk.
"""

from typing import List
import zlib
import flatbuffers
from google.protobuf import text_format
from litert_lm.python.tools import litertlm_core
from litert_lm.runtime.proto import llm_metadata_pb2
from litert_lm.schema.core import litertlm_header_schema_py_generated as schema

INT64_MAX = 9223372036854775807

INT64_MIN = -9223372036854775808


def _parse_metadata_value(value_str):
  """Converts a string from metadata into a bool, int, float, or string."""
  if value_str.lower() == "true":
    return True
  if value_str.lower() == "false":
    return False
  try:
    return int(value_str)
  except ValueError:
    pass
  try:
    return float(value_str)
  except ValueError:
    pass
  return value_str


def create_key_value_pair(builder, key, value):
  """Creates a FlatBuffers KeyValuePair."""
  key_offset = builder.CreateString(key)

  # 1. Create the inner value object FIRST.
  if isinstance(value, bool):
    schema.BoolStart(builder)
    schema.BoolAddValue(builder, value)
    value_offset = schema.BoolEnd(builder)
    value_type = schema.VData.Bool
  elif isinstance(value, int):
    if not INT64_MIN <= value <= INT64_MAX:
      raise ValueError(f"Integer value {value} is out of range for Int64")
    schema.Int64Start(builder)
    schema.Int64AddValue(builder, value)
    value_offset = schema.Int64End(builder)
    value_type = schema.VData.Int64
  elif isinstance(value, float):
    schema.DoubleStart(builder)
    schema.DoubleAddValue(builder, value)
    value_offset = schema.DoubleEnd(builder)
    value_type = schema.VData.Double
  else:  # Default to string
    value_offset_str = builder.CreateString(str(value))
    schema.StringValueStart(builder)
    schema.StringValueAddValue(builder, value_offset_str)
    value_offset = schema.StringValueEnd(builder)
    value_type = schema.VData.StringValue

  # 2. Now create the outer KeyValuePair object, using the offset of the
  #    value object we just created.
  schema.KeyValuePairStart(builder)
  schema.KeyValuePairAddKey(builder, key_offset)
  schema.KeyValuePairAddValueType(builder, value_type)
  schema.KeyValuePairAddValue(builder, value_offset)
  return schema.KeyValuePairEnd(builder)


def parse_metadata_string(metadata_str):
  """Parses the section_metadata string into a dictionary."""
  metadata_map = {}
  if not metadata_str:
    return metadata_map

  for section_part in metadata_str.split(";"):
    if not section_part:
      continue
    parts = section_part.split(":", 1)
    if len(parts) != 2:
      raise ValueError(f"Invalid section metadata format: {section_part}")
    section_name, kv_pairs_str = parts
    kv_map = {}
    if kv_pairs_str:
      for kv_str in kv_pairs_str.split(","):
        if "=" not in kv_str:
          raise ValueError(f"Invalid key-value pair: {kv_str}")
        key, value_str = kv_str.split("=", 1)
        kv_map[key] = _parse_metadata_value(value_str)
    metadata_map[section_name] = kv_map
  return metadata_map


def write_padding(f, block_size):
  """Writes zero padding to align to the next block size."""
  current_pos = f.tell()
  padding_needed = (block_size - (current_pos % block_size)) % block_size
  if padding_needed > 0:
    f.write(b"\0" * padding_needed)


def litertlm_write(
    output_path: str, input_files: List[str], section_metadata_str: str
):
  """Writes the LiteRT-LM file.

  This function takes a list of input files, combines them into a single
  LiteRT-LM file, and adds metadata. The LiteRT-LM file consists of:
    - A header with version information and metadata.
    - A series of sections, each containing data from an input file.
    - Padding to ensure block alignment.

  Args:
    output_path: The path to the output LiteRT-LM file.
    input_files: A list of input file paths.
    section_metadata_str: A string containing metadata for each section. The
      format is:
      "section_name1:key1=value1,key2=value2;section_name2:key3=value3"

  Raises:
    ValueError: If no input files are provided or header size exceeds limit.
  """
  if not input_files:
    raise ValueError("At least one input file must be provided.")

  metadata_map = parse_metadata_string(section_metadata_str)

  with open(output_path, "wb") as f:
    # 0. Write magic bytes and version
    f.write(b"LITERTLM")
    f.write(litertlm_core.LITERTLM_MAJOR_VERSION.to_bytes(4, "little"))
    f.write(litertlm_core.LITERTLM_MINOR_VERSION.to_bytes(4, "little"))
    f.write(litertlm_core.LITERTLM_PATCH_VERSION.to_bytes(4, "little"))

    # 1. Write zero pad until offset BLOCK_SIZE
    write_padding(f, litertlm_core.BLOCK_SIZE)

    # 2. Write the sections
    section_offsets = []
    section_types = []
    section_names = []

    for filename in input_files:
      section_type, section_name = litertlm_core.get_section_type_and_name(
          filename
      )
      section_types.append(section_type)
      section_names.append(section_name)

      start_offset = f.tell()
      with open(filename, "rb") as in_f:
        content = in_f.read()
        if section_type == schema.AnySectionDataType.LlmMetadataProto:
          ext = litertlm_core.get_file_extension(filename)
          if ext in (".pbtext", ".prototext"):
            metadata = llm_metadata_pb2.LlmMetadata()
            text_format.Parse(content.decode("utf-8"), metadata)
            f.write(metadata.SerializeToString())
          else:
            f.write(content)
        elif section_type == schema.AnySectionDataType.HF_Tokenizer_Zlib:
          uncompressed_size = len(content)
          compressed_content = zlib.compress(content)
          f.write(uncompressed_size.to_bytes(8, "little"))
          f.write(compressed_content)
        else:
          f.write(content)

      end_offset = f.tell()
      section_offsets.append((start_offset, end_offset))
      write_padding(f, litertlm_core.BLOCK_SIZE)

    # 3. Write the header
    builder = flatbuffers.Builder(1024)

    # System Metadata
    system_kv_pairs = [
        create_key_value_pair(builder, "author", "The ODML Authors")
    ]
    schema.SystemMetadataStartEntriesVector(builder, len(system_kv_pairs))
    for kv in reversed(system_kv_pairs):
      builder.PrependUOffsetTRelative(kv)
    entries_vec = builder.EndVector()
    schema.SystemMetadataStart(builder)
    schema.SystemMetadataAddEntries(builder, entries_vec)
    system_metadata_offset = schema.SystemMetadataEnd(builder)

    # Section Metadata
    section_objects = []
    for i, (start, end) in enumerate(section_offsets):
      section_name = section_names[i]
      section_kv_pairs = []
      if section_name in metadata_map:
        for key, value in metadata_map[section_name].items():
          section_kv_pairs.append(create_key_value_pair(builder, key, value))

      schema.SectionObjectStartItemsVector(builder, len(section_kv_pairs))
      for kv in reversed(section_kv_pairs):
        builder.PrependUOffsetTRelative(kv)
      items_vec = builder.EndVector()

      schema.SectionObjectStart(builder)
      schema.SectionObjectAddItems(builder, items_vec)
      schema.SectionObjectAddBeginOffset(builder, start)
      schema.SectionObjectAddEndOffset(builder, end)
      schema.SectionObjectAddDataType(builder, section_types[i])
      section_objects.append(schema.SectionObjectEnd(builder))

    schema.SectionMetadataStartObjectsVector(builder, len(section_objects))
    for obj in reversed(section_objects):
      builder.PrependUOffsetTRelative(obj)
    objects_vec = builder.EndVector()

    schema.SectionMetadataStart(builder)
    schema.SectionMetadataAddObjects(builder, objects_vec)
    section_metadata_offset = schema.SectionMetadataEnd(builder)

    # Root object
    schema.LiteRTLMMetaDataStart(builder)
    schema.LiteRTLMMetaDataAddSystemMetadata(builder, system_metadata_offset)
    schema.LiteRTLMMetaDataAddSectionMetadata(builder, section_metadata_offset)
    root = schema.LiteRTLMMetaDataEnd(builder)
    builder.Finish(root)

    header_data = builder.Output()

    f.seek(litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
    f.write(header_data)
    header_end_offset = f.tell()

    if header_end_offset > litertlm_core.BLOCK_SIZE:
      raise ValueError("Header size exceeds 16KB limit.")

    # 5. Finally, write the header end offset
    f.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
    f.write(header_end_offset.to_bytes(8, "little"))
