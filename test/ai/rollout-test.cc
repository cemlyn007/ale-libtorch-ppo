#include "ai/rollout.h"

#include <ale/ale_interface.hpp>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include "gtest/gtest.h"

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

  // Small CPU rollout whose selector always picks action 0; logits are sized
  // from the ROM's minimal action set so Buffer::add accepts them.
  ai::rollout::Rollout construct_rollout(size_t num_workers) {
    ale::ALEInterface ale;
    ale.loadROM(rom_path_);
    const auto action_size =
        static_cast<int64_t>(ale.getMinimalActionSet().size());
    return ai::rollout::Rollout(
        rom_path_, kTotalEnvironments, kHorizon, /*max_steps=*/1000,
        /*frame_stack=*/4, /*grayscale=*/true,
        [action_size](
            const torch::Tensor &observations) -> ai::rollout::ActionResult {
          const int64_t batch = observations.size(0);
          return {torch::zeros({batch}, torch::kLong),
                  torch::zeros({batch, action_size}), torch::zeros({batch})};
        },
        /*gae_discount=*/0.99f, /*gae_lambda=*/0.95f,
        torch::Device(torch::kCPU), /*seed=*/42, num_workers,
        /*worker_batch_size=*/1, /*frame_skip=*/4);
  }

  static constexpr size_t kTotalEnvironments = 2;
  static constexpr size_t kHorizon = 4;

  std::filesystem::path rom_path_;
};

// A fully-constructed Rollout destroyed before rollout() ever runs must shut
// down cleanly: action_result_ is only assigned inside rollout(), and fill_
// on the undefined tensor would throw inside the noexcept destructor and
// terminate the process.
TEST_F(RolloutTest, DestroyBeforeFirstRollout) {
  {
    auto rollout = construct_rollout(/*num_workers=*/1);
  }
  SUCCEED();
}

// With zero workers nothing ever services the action queue, so the first
// step_all() would block forever; the constructor must reject it instead.
TEST_F(RolloutTest, ZeroWorkersThrows) {
  EXPECT_THROW(construct_rollout(/*num_workers=*/0), std::invalid_argument);
}

// Destruction after a completed rollout exercises the defined-tensor branch
// of the destructor as well as the normal worker shutdown.
TEST_F(RolloutTest, RolloutThenDestroy) {
  auto rollout = construct_rollout(/*num_workers=*/1);
  const auto result = rollout.rollout();
  EXPECT_EQ(result.batch.observations.size(0),
            static_cast<int64_t>(kTotalEnvironments));
  EXPECT_EQ(result.batch.observations.size(1), static_cast<int64_t>(kHorizon));
  EXPECT_GT(result.log.steps, 0u);
}
