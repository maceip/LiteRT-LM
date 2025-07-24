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

"""Library for inspecting the contents of a LiteRT-LM file."""

import struct
from google.protobuf import text_format
from litert_lm.python.tools import litertlm_core
from litert_lm.runtime.proto import llm_metadata_pb2
from litert_lm.schema.core import litertlm_header_schema_py_generated as schema

# --- ANSI Escape Code Definitions ---
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
# --- Indentation Constants ---
INDENT_SPACES = 2


def print_boxed_title(os, title, box_width=50):
  """Prints a title surrounded by an ASCII box."""
  top_bottom = "+" + "-" * (box_width - 2) + "+"
  padding_left = (box_width - 2 - len(title)) // 2
  padding_right = box_width - 2 - len(title) - padding_left
  middle = "|" + " " * padding_left + title + " " * padding_right + "|"
  os.write(f"{top_bottom}\n{middle}\n{top_bottom}\n")


def print_key_value_pair(kvp, output_stream, indent_level):
  """Prints a formatted KeyValuePair."""
  indent_str = " " * (indent_level * INDENT_SPACES)
  if not kvp:
    output_stream.write(f"{indent_str}KeyValuePair: nullptr\n")
    return

  use_color = hasattr(output_stream, "isatty") and output_stream.isatty()
  bold = ANSI_BOLD if use_color else ""
  reset = ANSI_RESET if use_color else ""

  key_bytes = kvp.Key()
  key = key_bytes.decode("utf-8") if key_bytes is not None else None
  output_stream.write(f"{indent_str}{bold}Key{reset}: {key}, ")

  value_type = kvp.ValueType()
  union_table = kvp.Value()

  if union_table is None:
    output_stream.write(f"{bold}Value{reset}: <null>\n")
    return

  if value_type == schema.VData.StringValue:
    value_obj = schema.StringValue()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    value_bytes = value_obj.Value()
    value = value_bytes.decode("utf-8") if value_bytes is not None else None
    output_stream.write(f"{bold}Value{reset} (String): {value}\n")
  elif value_type == schema.VData.UInt8:
    value_obj = schema.UInt8()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt8): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int8:
    value_obj = schema.Int8()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int8): {value_obj.Value()}\n")
  elif value_type == schema.VData.UInt16:
    value_obj = schema.UInt16()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt16): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int16:
    value_obj = schema.Int16()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int16): {value_obj.Value()}\n")
  elif value_type == schema.VData.UInt32:
    value_obj = schema.UInt32()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt32): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int32:
    value_obj = schema.Int32()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int32): {value_obj.Value()}\n")
  elif value_type == schema.VData.UInt64:
    value_obj = schema.UInt64()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (UInt64): {value_obj.Value()}\n")
  elif value_type == schema.VData.Int64:
    value_obj = schema.Int64()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(f"{bold}Value{reset} (Int64): {value_obj.Value()}\n")
  elif value_type == schema.VData.Double:
    value_obj = schema.Double()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(
        f"{bold}Value{reset} (Double): {value_obj.Value():.4f}\n"
    )
  elif value_type == schema.VData.Bool:
    value_obj = schema.Bool()
    value_obj.Init(union_table.Bytes, union_table.Pos)
    output_stream.write(
        f"{bold}Value{reset} (Bool): {bool(value_obj.Value())}\n"
    )
  else:
    output_stream.write(f"{bold}Value{reset} (Unknown Type)\n")


def read_litertlm_header(file_path, output_stream):
  """Reads the header of a LiteRT-LM file and returns the metadata."""
  with open(file_path, "rb") as f:
    magic = f.read(8)
    if magic != b"LITERTLM":
      raise ValueError(f"Invalid magic number: {magic}")

    major, minor, patch = struct.unpack("<III", f.read(12))
    output_stream.write(f"LiteRT-LM Version: {major}.{minor}.{patch}\n\n")

    f.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
    header_end_offset = struct.unpack("<Q", f.read(8))[0]

    f.seek(litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
    header_data = f.read(
        header_end_offset - litertlm_core.HEADER_BEGIN_BYTE_OFFSET
    )

    metadata = schema.LiteRTLMMetaData.GetRootAs(header_data, 0)
    return metadata


def peek_litertlm_file(litertlm_path, output_stream):
  """Reads and prints information from a LiteRT-LM file."""
  metadata = read_litertlm_header(litertlm_path, output_stream)
  with open(litertlm_path, "rb") as f:

    # Print System Metadata
    system_metadata = metadata.SystemMetadata()
    print_boxed_title(output_stream, "System Metadata")
    if system_metadata and system_metadata.EntriesLength() > 0:
      for i in range(system_metadata.EntriesLength()):
        print_key_value_pair(system_metadata.Entries(i), output_stream, 1)
    else:
      output_stream.write(" " * INDENT_SPACES + "No system metadata entries.\n")
    output_stream.write("\n")

    # Print Section Metadata
    section_metadata = metadata.SectionMetadata()
    num_sections = section_metadata.ObjectsLength() if section_metadata else 0
    print_boxed_title(output_stream, f"Sections ({num_sections})")

    if num_sections == 0 or section_metadata is None:
      output_stream.write(" " * INDENT_SPACES + "<None>\n")
    else:
      use_color = hasattr(output_stream, "isatty") and output_stream.isatty()
      bold = ANSI_BOLD if use_color else ""
      reset = ANSI_RESET if use_color else ""
      for i in range(num_sections):
        sec_obj = section_metadata.Objects(i)
        output_stream.write(f"\n{bold}Section {i}:{reset}\n")
        output_stream.write(" " * INDENT_SPACES + "Items:\n")
        if sec_obj is None:
          output_stream.write(" " * INDENT_SPACES + "<None>\n")
          continue

        # Print the items in the section.
        if sec_obj.ItemsLength() > 0:
          for j in range(sec_obj.ItemsLength()):
            print_key_value_pair(sec_obj.Items(j), output_stream, 2)
        else:
          output_stream.write(" " * (2 * INDENT_SPACES) + "<None>\n")

        output_stream.write(
            f"{' ' * INDENT_SPACES}Begin Offset: {sec_obj.BeginOffset()}\n"
        )
        output_stream.write(
            f"{' ' * INDENT_SPACES}End Offset:   {sec_obj.EndOffset()}\n"
        )
        output_stream.write(
            f"{' ' * INDENT_SPACES}Data Type:    "
            f"{litertlm_core.any_section_data_type_to_string(sec_obj.DataType())}\n"
        )

        if sec_obj.DataType() == schema.AnySectionDataType.LlmMetadataProto:
          f.seek(sec_obj.BeginOffset())
          proto_data = f.read(sec_obj.EndOffset() - sec_obj.BeginOffset())
          llm_metadata = llm_metadata_pb2.LlmMetadata()
          llm_metadata.ParseFromString(proto_data)
          output_stream.write(
              f"{' ' * INDENT_SPACES}<<<<<<<< start of LlmMetadata\n"
          )
          debug_str = text_format.MessageToString(llm_metadata)
          for line in debug_str.splitlines():
            output_stream.write(f"{' ' * (INDENT_SPACES * 2)}{line}\n")
          output_stream.write(
              f"{' ' * INDENT_SPACES}>>>>>>>> end of LlmMetadata\n"
          )
        output_stream.write("\n")
