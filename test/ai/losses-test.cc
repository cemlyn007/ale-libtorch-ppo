#include "ai/ppo/losses.h"

#include <torch/torch.h>

#include <cmath>

#include "gtest/gtest.h"

namespace {

// A small, valid set of PPO inputs for n transitions over a actions. Uniform
// (normalized) log probabilities keep every derived quantity finite, so the
// only way the loss can be non-finite is the divide-by-zero under test.
struct Inputs {
  torch::Tensor log_probabilities;
  torch::Tensor old_log_probabilities;
  torch::Tensor actions;
  torch::Tensor advantages;
  torch::Tensor values;
  torch::Tensor returns;
};

Inputs make_inputs(int64_t n, int64_t a) {
  Inputs in;
  in.log_probabilities =
      ai::ppo::losses::normalize_logits(torch::zeros({n, a}));
  in.old_log_probabilities = in.log_probabilities.clone();
  in.actions = torch::zeros({n}, torch::kLong);
  in.advantages = torch::ones({n});
  in.values = torch::zeros({n});
  in.returns = torch::ones({n});
  return in;
}

ai::ppo::losses::Metrics compute(const Inputs &in, const torch::Tensor &masks) {
  return ai::ppo::losses::compute(
      in.log_probabilities, in.old_log_probabilities, in.actions, in.advantages,
      in.values, in.returns, masks, /*clip_param=*/0.2,
      /*value_loss_coef=*/0.5, /*entropy_coef=*/0.01);
}

}  // namespace

TEST(LossesTest, FullyMaskedMinibatchYieldsZeroNotNaN) {
  auto in = make_inputs(/*n=*/4, /*a=*/2);
  // Every transition masked out -> masks.sum() == 0 would be a 0/0 = NaN.
  auto metrics = compute(in, torch::zeros({4}, torch::kBool));
  float loss = metrics.loss.item<float>();
  EXPECT_TRUE(std::isfinite(loss)) << "loss was " << loss;
  EXPECT_FLOAT_EQ(loss, 0.0f);
}

TEST(LossesTest, MaskGuardIsNoOpWhenNothingMasked) {
  auto in = make_inputs(/*n=*/4, /*a=*/2);
  auto metrics = compute(in, torch::ones({4}, torch::kBool));
  // With nothing masked the loss is the plain mean of the per-transition total
  // losses; clamping the denominator must not perturb it.
  EXPECT_NEAR(metrics.loss.item<float>(),
              metrics.total_losses.mean().item<float>(), 1e-6);
}
