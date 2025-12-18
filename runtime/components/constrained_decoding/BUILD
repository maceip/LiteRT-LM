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

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "bitmap",
    hdrs = ["bitmap.h"],
)

cc_library(
    name = "constraint",
    hdrs = ["constraint.h"],
    deps = [
        ":bitmap",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "constraint_provider",
    hdrs = ["constraint_provider.h"],
    deps = [
        ":constraint",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "constrained_decoder",
    srcs = ["constrained_decoder.cc"],
    hdrs = ["constrained_decoder.h"],
    deps = [
        ":constraint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "fake_constraint",
    srcs = ["fake_constraint.cc"],
    hdrs = ["fake_constraint.h"],
    deps = [
        ":bitmap",
        ":constraint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

alias(
    name = "gemma_model_constraint_provider_lib",
    actual = ":gemma_model_constraint_provider_shared_lib",
)

cc_import(
    name = "gemma_model_constraint_provider_dll_import",
    interface_library = "//prebuilt/windows_x86_64:libGemmaModelConstraintProvider.lib",
    shared_library = "//prebuilt/windows_x86_64:libGemmaModelConstraintProvider.dll",
    target_compatible_with = ["@platforms//os:windows"],
)

cc_library(
    name = "gemma_model_constraint_provider_shared_lib",
    srcs = select({
        "@platforms//os:android": ["//prebuilt/android_arm64:libGemmaModelConstraintProvider.so"],
        "//build_config:linux_x86_64": ["//prebuilt/linux_x86_64:libGemmaModelConstraintProvider.so"],
        "//build_config:linux_arm64": ["//prebuilt/linux_arm64:libGemmaModelConstraintProvider.so"],
        "//build_config:ios_sim_arm64": ["//prebuilt/ios_sim_arm64:libGemmaModelConstraintProvider.dylib"],
        "//build_config:ios_arm64": ["//prebuilt/ios_arm64:libGemmaModelConstraintProvider.dylib"],
        "@platforms//os:macos": ["//prebuilt/macos_arm64:libGemmaModelConstraintProvider.dylib"],
        "@platforms//os:windows": [],
        "//conditions:default": [],
    }),
    hdrs = [
        "gemma_model_constraint_provider.h",
    ],
    linkopts = select({
        "@platforms//os:ios": ["-Wl,-rpath,@loader_path"],
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        ":constraint",
        ":constraint_provider",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "@sentencepiece//:sentencepiece_processor",
    ] + select({
        "@platforms//os:windows": [":gemma_model_constraint_provider_dll_import"],
        "//conditions:default": [],
    }),
)
