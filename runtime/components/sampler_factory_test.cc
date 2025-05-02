#include "runtime/components/sampler_factory.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

TEST(SamplerFactoryTest, CreateSamplerForCpuWorksCorrectly) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  ASSERT_OK_AND_ASSIGN(
      auto sampler, CreateSampler(Backend::CPU,
                                  /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_NE(sampler, nullptr);
  // Make sure the factory creates the correct sampler.
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithUnsupportedSamplerTypeFails) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_K);
  auto result = CreateSampler(Backend::CPU,
                              /*batch_size=*/1, std::move(sampler_params));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(result.status().message(), HasSubstr("not implemented yet."));
}

TEST(SamplerFactoryTest,
     CreateSamplerForCpuWithUnspecifiedSamplerTypeReturnsNullptr) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_EQ(sampler, nullptr);
}

}  // namespace
}  // namespace litert::lm
