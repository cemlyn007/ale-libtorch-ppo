#include "ai/rollout.h"

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include "gtest/gtest.h"

namespace {

// Breakout's minimal action set: NOOP, FIRE, RIGHT, LEFT.
constexpr int64_t kBreakoutActionSetSize = 4;

// Always selects action 0 (NOOP), with the shapes Rollout/Buffer expect.
ai::rollout::ActionResult zero_action_selector(
    const torch::Tensor &observations) {
  const int64_t n = observations.size(0);
  return {torch::zeros({n}, torch::kLong),
          torch::zeros({n, kBreakoutActionSetSize}), torch::zeros({n})};
}

}  // namespace

class RolloutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *test_srcdir = std::getenv("TEST_SRCDIR");
    GTEST_ASSERT_TRUE(test_srcdir != nullptr)
        << "TEST_SRCDIR environment variable is not set";
    rom_path_ =
        std::filesystem::path(test_srcdir) / "_main" / "roms" / "breakout.bin";
    GTEST_ASSERT_TRUE(std::filesystem::exists(rom_path_))
        << "ROM file not found: " << rom_path_;
  }

  void construct_rollout(size_t total_environments, size_t worker_batch_size) {
    ai::rollout::Rollout rollout(
        rom_path_, total_environments, /*horizon=*/1, /*max_steps=*/1000,
        /*frame_stack=*/1, /*grayscale=*/true, zero_action_selector,
        /*gae_discount=*/0.99f, /*gae_lambda=*/0.95f, torch::kCPU, /*seed=*/42,
        /*num_workers=*/1, worker_batch_size, /*frame_skip=*/4);
  }

  std::filesystem::path rom_path_;
};

// A worker batch size that does not divide the environment count would leave a
// final sub-batch no worker can ever pop, deadlocking step_all(); the
// constructor must reject it up front.
TEST_F(RolloutTest, ThrowsWhenWorkerBatchSizeDoesNotDivideTotalEnvironments) {
  EXPECT_THROW(construct_rollout(/*total_environments=*/4,
                                 /*worker_batch_size=*/3),
               std::invalid_argument);
}

TEST_F(RolloutTest, ThrowsWhenWorkerBatchSizeIsZero) {
  EXPECT_THROW(construct_rollout(/*total_environments=*/4,
                                 /*worker_batch_size=*/0),
               std::invalid_argument);
}

// Positive control: a dividing batch size constructs and completes a rollout.
TEST_F(RolloutTest, DivisibleWorkerBatchSizeRollsOut) {
  ai::rollout::Rollout rollout(
      rom_path_, /*total_environments=*/2, /*horizon=*/2, /*max_steps=*/1000,
      /*frame_stack=*/1, /*grayscale=*/true, zero_action_selector,
      /*gae_discount=*/0.99f, /*gae_lambda=*/0.95f, torch::kCPU, /*seed=*/42,
      /*num_workers=*/1, /*worker_batch_size=*/2, /*frame_skip=*/4);
  const auto result = rollout.rollout();
  // The first step of each env is its episode-start reset, which is not
  // counted; only the second step is.
  EXPECT_EQ(result.log.steps, 2u);
  EXPECT_EQ(result.log.episodes, 0u);
}
