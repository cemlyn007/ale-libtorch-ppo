#include "training/bandit.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "training/config.h"

namespace {

// A fully-populated, internally-consistent Config to mutate in tests. validate()
// only inspects the async/deterministic pair and the batch geometry, but
// sample_arms copies the whole struct, so every field is initialised.
training::Config valid_base() {
  training::Config c{};
  c.total_environments = 4096;
  c.hidden_size = 512;
  c.horizon = 5;
  c.max_steps = 64800;
  c.frame_stack = 4;
  c.learning_rate = 2.5e-4;
  c.clip_param = 0.2f;
  c.value_loss_coef = 0.4f;
  c.entropy_coef = 0.01f;
  c.num_epochs = 2;
  c.num_mini_batches = 16;
  c.shuffle_mini_batches = true;
  c.gae_discount = 0.99f;
  c.gae_lambda = 0.95f;
  c.max_gradient_norm = 0.5f;
  c.num_rollouts = 1000;
  c.num_workers = 32;
  c.worker_batch_size = 8;
  c.pipeline_groups = 2;
  c.frame_skip = 4;
  c.max_return = 864.0f;
  c.record_observation = false;
  c.record_video = false;
  c.cuda_graph = false;
  c.async_update = false;
  c.deterministic = false;
  c.checkpoint_interval = 0;
  training::reconcile(c);  // mini_batch_size = 5 * 4096 / 16 = 1280
  return c;
}

// A choice spec that writes `field` by name, mirroring load_search_space().
training::bandit::ParamSpec choice(std::string field,
                                   std::vector<double> choices) {
  training::bandit::ParamSpec spec;
  spec.name = field;
  spec.distribution = training::bandit::Distribution::kChoice;
  spec.choices = std::move(choices);
  spec.apply = [field](training::Config &c, double v) {
    training::apply_field(c, field, v);
  };
  return spec;
}

training::bandit::ParamSpec int_uniform(std::string field, double low,
                                        double high) {
  training::bandit::ParamSpec spec;
  spec.name = field;
  spec.distribution = training::bandit::Distribution::kIntUniform;
  spec.low = low;
  spec.high = high;
  spec.apply = [field](training::Config &c, double v) {
    training::apply_field(c, field, v);
  };
  return spec;
}

TEST(Reconcile, DerivesMiniBatchSizeFromBatchShape) {
  training::Config c = valid_base();
  c.total_environments = 8192;
  c.horizon = 7;
  c.num_mini_batches = 32;
  training::reconcile(c);
  EXPECT_EQ(c.mini_batch_size, 8192 * 7 / 32);  // 1792
  EXPECT_NO_THROW(training::validate(c));
}

TEST(Validate, AcceptsConsistentBase) {
  EXPECT_NO_THROW(training::validate(valid_base()));
}

TEST(Validate, RejectsInconsistentMiniBatchSize) {
  training::Config c = valid_base();
  c.mini_batch_size += 1;  // no longer == size / num_mini_batches
  EXPECT_THROW(training::validate(c), std::invalid_argument);
}

TEST(Validate, RejectsNumMiniBatchesNotDividingRollout) {
  training::Config c = valid_base();
  c.num_mini_batches = 7;  // 20480 % 7 != 0
  training::reconcile(c);
  EXPECT_THROW(training::validate(c), std::invalid_argument);
}

TEST(Validate, RejectsWorkerBatchNotDividingGroup) {
  training::Config c = valid_base();  // group = 4096 / 2 = 2048
  c.worker_batch_size = 3;            // 2048 % 3 != 0
  EXPECT_THROW(training::validate(c), std::invalid_argument);
}

TEST(Validate, RejectsPipelineGroupsNotDividingEnvs) {
  training::Config c = valid_base();
  c.pipeline_groups = 3;  // 4096 % 3 != 0
  EXPECT_THROW(training::validate(c), std::invalid_argument);
}

// The property the whole change exists to guarantee: sampling the coupled batch
// knobs (plus a free horizon) yields only internally-consistent arms.
TEST(SampleArms, EveryArmPassesValidateWithBatchGeometry) {
  training::bandit::SearchSpace space;
  space.push_back(choice("total_environments", {2048, 4096, 8192}));
  space.push_back(choice("num_mini_batches", {8, 16, 32}));
  space.push_back(choice("pipeline_groups", {1, 2, 4}));
  space.push_back(choice("worker_batch_size", {4, 8, 16}));
  space.push_back(int_uniform("horizon", 4, 16));

  const training::Config base = valid_base();
  const std::vector<training::bandit::Arm> arms =
      training::bandit::sample_arms(base, 500, /*seed=*/123, space);
  ASSERT_EQ(arms.size(), 500u);
  for (const training::bandit::Arm &arm : arms) {
    EXPECT_NO_THROW(training::validate(arm.config))
        << "arm " << arm.id << " total_env=" << arm.config.total_environments
        << " horizon=" << arm.config.horizon
        << " nmb=" << arm.config.num_mini_batches
        << " mb=" << arm.config.mini_batch_size
        << " pg=" << arm.config.pipeline_groups
        << " wbs=" << arm.config.worker_batch_size;
    // mini_batch_size really is the derived value, not a stale base copy.
    EXPECT_EQ(static_cast<std::size_t>(arm.config.mini_batch_size) *
                  static_cast<std::size_t>(arm.config.num_mini_batches),
              arm.config.horizon * arm.config.total_environments);
  }
}

}  // namespace
