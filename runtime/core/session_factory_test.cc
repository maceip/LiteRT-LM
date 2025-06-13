#include "runtime/core/session_factory.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

class FakeTokenizer : public Tokenizer {
 public:
  FakeTokenizer() = default;

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    return std::vector<int>{1, 2, 3};
  }

  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override {
    return "fake_text";
  }

  absl::StatusOr<int> BosId() const override { return 2; }

  absl::StatusOr<int> EosId() const override { return 1; }
};

TEST(SessionFactoryTest, InitializeSession) {
  FakeTokenizer tokenizer;
  std::vector<std::vector<int>> stop_token_ids = {{1}, {2}};
  std::vector<std::vector<int>> dummy_tokens = {{0}};
  FakeLlmExecutor executor(256, dummy_tokens, dummy_tokens);
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  ThreadPool worker_thread_pool("testpool", /*max_num_threads=*/1);
  auto session =
      InitializeSession(&executor, &tokenizer, session_config,
                        /*benchmark_info=*/std::nullopt, &worker_thread_pool);
  EXPECT_OK(session);
}

}  // namespace
}  // namespace litert::lm
