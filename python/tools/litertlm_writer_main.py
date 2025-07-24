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

r"""This tool is used to create a LiteRT-LM file from a set of input files.

For example tokenizer, tflite model, llm parameters and metadata.

Example usage:

  python litertlm_writer_main.py \
    --output_path=/path/to/output.litertlm \
    /path/to/llm_metadata.pbtext \
    /path/to/tokenizer.spiece \
    /path/to/model.tflite \
    --section_metadata="llm_metadata:key1=101,key2=value2;tokenizer:key1=value1,key2=value2;tflite:key3=123"

Note: the order of the input files is important, and should match the order
specified in the section_metadata argument.

"""

import argparse
from litert_lm.python.tools import litertlm_writer


def main():
  """Parses command-line arguments and runs the litertlm_writer tool."""
  parser = argparse.ArgumentParser(
      description="Create a LiteRT-LM file from input files and metadata."
  )
  parser.add_argument(
      "--output_path",
      type=str,
      required=True,
      help="The path for the output LiteRT-LM file.",
  )
  parser.add_argument(
      "--section_metadata",
      type=str,
      default="",
      help=(
          "Metadata for sections in the format "
          "'section_name:key1=value1,key2=value2;...'."
      ),
  )
  parser.add_argument(
      "input_files",
      nargs="+",
      help="Paths to the input files (e.g., tokenizer, model).",
  )

  args = parser.parse_args()

  try:
    litertlm_writer.litertlm_write(
        args.output_path, args.input_files, args.section_metadata
    )
    print(
        "🎂 LiteRT-LM file successfully created! Output is at"
        f" {args.output_path}"
    )
  except (ValueError, FileNotFoundError) as e:
    print(f"Error creating LiteRT-LM file: {e}")
    exit(1)


if __name__ == "__main__":
  main()
