#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/sampler_factory.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

TEST(SamplerFactoryFailedDlopenTest,
     CreateSamplerForGpuFallsBackToCpuIfUnavailable) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_env,
                               litert::Environment::Create({}));

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  auto result =
      CreateSampler(Backend::GPU, /*batch_size=*/1, std::move(sampler_params),
                    litert_env.Get(), /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt);

  // Should fall back to CPU sampler.
  ASSERT_OK(result);
  auto sampler = std::move(result.value());
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
}

}  // namespace
}  // namespace litert::lm
