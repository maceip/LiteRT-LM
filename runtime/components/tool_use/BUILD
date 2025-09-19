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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

cc_library(
    name = "python_tool_format_utils",
    srcs = ["python_tool_format_utils.cc"],
    hdrs = ["python_tool_format_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "python_tool_format_utils_test",
    srcs = ["python_tool_format_utils_test.cc"],
    deps = [
        ":python_tool_format_utils",
        "@com_google_googletest//:gtest_main",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)
