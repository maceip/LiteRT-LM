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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_STATUS_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_STATUS_UTIL_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert

namespace litert::lm {

// Converts litert::Expected<T> to absl::StatusOr<T>.
absl::Status ToAbslStatus(const ::litert::Error& error);

// For Expected<void>.
absl::Status ToAbslStatus(::litert::Expected<void> expected);

// Converts absl::StatusOr<T> from litert::Expected<T>.
template <typename T>
absl::StatusOr<T> ToAbslStatus(::litert::Expected<T> expected) {
  if (expected.HasValue()) {
    return std::move(*expected);
  }
  return ToAbslStatus(expected.Error());
}

}  // namespace litert::lm

// Same as LITERT_ASSIGN_OR_RETURN, but returns absl::StatusOr<T> instead of
// litert::Expected<T>.
#define LITERT_ASSIGN_OR_RETURN_ABSL(DECL, ...)             \
  LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD(                  \
      (DECL, __VA_ARGS__, LITERT_ASSIGN_OR_RETURN_HELPER_3, \
       LITERT_ASSIGN_OR_RETURN_ABSL_HELPER_2))(             \
      _CONCAT_NAME(expected_value_or_error_, __LINE__), DECL, __VA_ARGS__)

#define LITERT_ASSIGN_OR_RETURN_ABSL_HELPER_2(TMP_VAR, DECL, EXPR) \
  LITERT_ASSIGN_OR_RETURN_HELPER_3(                                \
      TMP_VAR, DECL, EXPR, ::litert::lm::ToAbslStatus(TMP_VAR.Error()))

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_STATUS_UTIL_H_
