#include "ai/rollout.h"

#include <torch/torch.h>

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
  // from the ROM's minimal action set so the buffer accepts them. The selector
  // is deterministic, so trajectories must not depend on pipeline grouping.
  ai::rollout::Rollout construct_rollout(size_t total_environments,
                                         size_t num_workers,
                                         size_t worker_batch_size,
                                         size_t pipeline_groups = 1,
                                         size_t num_buffers = 1) {
    ale::ALEInterface ale;
    ale.loadROM(rom_path_);
    const auto action_size =
        static_cast<int64_t>(ale.getMinimalActionSet().size());
    return ai::rollout::Rollout(
        rom_path_, total_environments, kHorizon, /*max_steps=*/1000,
        /*frame_stack=*/4, /*grayscale=*/true,
        [action_size](
            const torch::Tensor &observations) -> ai::rollout::ActionResult {
          const int64_t batch = observations.size(0);
          return {torch::zeros({batch}, torch::kLong),
                  torch::zeros({batch, action_size}), torch::zeros({batch})};
        },
        /*gae_discount=*/0.99f, /*gae_lambda=*/0.95f,
        torch::Device(torch::kCPU), /*seed=*/42, num_workers, worker_batch_size,
        /*frame_skip=*/4, /*max_return=*/0.0f,
        /*video_path=*/std::nullopt, /*record_observation=*/false,
        pipeline_groups, num_buffers);
  }

  static constexpr size_t kTotalEnvironments = 2;
  static constexpr size_t kHorizon = 4;

  std::filesystem::path rom_path_;
};

// A worker batch size that does not divide the environment count would leave a
// final sub-batch no worker can ever pop, deadlocking rollout(); the
// constructor must reject it up front.
TEST_F(RolloutTest, ThrowsWhenWorkerBatchSizeDoesNotDivideTotalEnvironments) {
  EXPECT_THROW(construct_rollout(/*total_environments=*/4, /*num_workers=*/1,
                                 /*worker_batch_size=*/3),
               std::invalid_argument);
}

TEST_F(RolloutTest, ThrowsWhenWorkerBatchSizeIsZero) {
  EXPECT_THROW(construct_rollout(/*total_environments=*/4, /*num_workers=*/1,
                                 /*worker_batch_size=*/0),
               std::invalid_argument);
}

// With zero workers nothing ever services the action queue, so the first
// rollout() would block forever; the constructor must reject it instead.
TEST_F(RolloutTest, ZeroWorkersThrows) {
  EXPECT_THROW(construct_rollout(kTotalEnvironments, /*num_workers=*/0,
                                 /*worker_batch_size=*/1),
               std::invalid_argument);
}

// A fully-constructed Rollout destroyed before rollout() ever runs must shut
// down cleanly: workers blocked on the action queue have to be woken onto the
// reset path and joined without ever reading the (unstaged) actions.
TEST_F(RolloutTest, DestroyBeforeFirstRollout) {
  {
    auto rollout = construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                     /*worker_batch_size=*/1);
  }
  SUCCEED();
}

// Destruction after a completed rollout exercises worker shutdown with envs
// mid-episode rather than at their initial reset.
TEST_F(RolloutTest, RolloutThenDestroy) {
  auto rollout = construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                   /*worker_batch_size=*/1);
  const auto result = rollout.rollout();
  EXPECT_EQ(result.batch.observations.size(0),
            static_cast<int64_t>(kTotalEnvironments));
  EXPECT_EQ(result.batch.observations.size(1), static_cast<int64_t>(kHorizon));
  EXPECT_GT(result.log.steps, 0u);
}

// Positive control for the divisibility check: a batch size that spans all
// environments at once constructs and completes a rollout, with exact step
// accounting (each env's first add is its episode-start reset, not counted).
TEST_F(RolloutTest, DivisibleWorkerBatchSizeRollsOut) {
  auto rollout = construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                   /*worker_batch_size=*/kTotalEnvironments);
  const auto result = rollout.rollout();
  EXPECT_EQ(result.log.steps, (kHorizon - 1) * kTotalEnvironments);
  EXPECT_EQ(result.log.episodes, 0u);
}

// The constructor must reject group configurations that would starve
// Queue::pop and hang: zero groups, groups that do not divide the env count,
// and a worker batch that does not divide the group size.
TEST_F(RolloutTest, InvalidPipelineGroupingThrows) {
  EXPECT_THROW(construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                 /*worker_batch_size=*/1,
                                 /*pipeline_groups=*/0),
               std::invalid_argument);
  EXPECT_THROW(construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                 /*worker_batch_size=*/1,
                                 /*pipeline_groups=*/3),
               std::invalid_argument);
  EXPECT_THROW(construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                 /*worker_batch_size=*/2,
                                 /*pipeline_groups=*/2),
               std::invalid_argument);
}

// A zero buffer count could never store a rollout; the constructor must
// reject it up front like the other degenerate sizes.
TEST_F(RolloutTest, ZeroBuffersThrows) {
  EXPECT_THROW(construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                 /*worker_batch_size=*/1, /*pipeline_groups=*/1,
                                 /*num_buffers=*/0),
               std::invalid_argument);
}

// The default single buffer hands back the same storage every rollout — the
// contract the synchronous training loop relies on (and the baseline for the
// double-buffered test below).
TEST_F(RolloutTest, SingleBufferReusesStorage) {
  auto rollout = construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                   /*worker_batch_size=*/1);
  const auto first = rollout.rollout();
  const auto second = rollout.rollout();
  EXPECT_EQ(first.batch.observations.data_ptr(),
            second.batch.observations.data_ptr());
}

// With two buffers, consecutive rollouts must land in alternating storage and
// leave the previous batch intact while the next one is collected — the
// guarantee the async update trains on. The third rollout wraps around to the
// first buffer again.
TEST_F(RolloutTest, DoubleBufferingPreservesPreviousBatch) {
  auto rollout = construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                   /*worker_batch_size=*/1,
                                   /*pipeline_groups=*/1, /*num_buffers=*/2);
  const auto first = rollout.rollout();
  const auto first_observations = first.batch.observations.clone();
  const auto first_advantages = first.batch.advantages.clone();
  const auto second = rollout.rollout();
  EXPECT_NE(first.batch.observations.data_ptr(),
            second.batch.observations.data_ptr());
  EXPECT_NE(first.batch.advantages.data_ptr(),
            second.batch.advantages.data_ptr());
  EXPECT_TRUE(torch::equal(first.batch.observations, first_observations));
  EXPECT_TRUE(torch::equal(first.batch.advantages, first_advantages));
  const auto third = rollout.rollout();
  EXPECT_EQ(first.batch.observations.data_ptr(),
            third.batch.observations.data_ptr());
}

// With a deterministic selector, grouping must not change trajectories: the
// pipelined loop (two groups) and the classic synchronous loop (one group)
// must produce bit-identical batches and logs, rollout after rollout. One
// worker keeps result-queue order deterministic so the logs compare exactly.
TEST_F(RolloutTest, PipelinedMatchesSynchronous) {
  auto synchronous = construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                       /*worker_batch_size=*/1);
  auto pipelined = construct_rollout(kTotalEnvironments, /*num_workers=*/1,
                                     /*worker_batch_size=*/1,
                                     /*pipeline_groups=*/2);
  for (int iteration = 0; iteration < 3; ++iteration) {
    const auto expected = synchronous.rollout();
    const auto actual = pipelined.rollout();
    EXPECT_TRUE(
        torch::equal(expected.batch.observations, actual.batch.observations));
    EXPECT_TRUE(torch::equal(expected.batch.actions, actual.batch.actions));
    EXPECT_TRUE(torch::equal(expected.batch.rewards, actual.batch.rewards));
    EXPECT_TRUE(torch::equal(expected.batch.masks, actual.batch.masks));
    EXPECT_TRUE(torch::equal(expected.batch.logits, actual.batch.logits));
    EXPECT_TRUE(torch::equal(expected.batch.values, actual.batch.values));
    EXPECT_TRUE(
        torch::equal(expected.batch.advantages, actual.batch.advantages));
    EXPECT_TRUE(torch::equal(expected.batch.returns, actual.batch.returns));
    EXPECT_EQ(expected.log.steps, actual.log.steps);
    EXPECT_EQ(expected.log.episodes, actual.log.episodes);
    EXPECT_EQ(expected.log.episode_returns, actual.log.episode_returns);
    EXPECT_EQ(expected.log.episode_lengths, actual.log.episode_lengths);
    EXPECT_EQ(expected.log.game_returns, actual.log.game_returns);
    EXPECT_EQ(expected.log.game_lengths, actual.log.game_lengths);
  }
}
