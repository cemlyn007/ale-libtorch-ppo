#include "ai/environment/environment.h"
#include "gtest/gtest.h"
#include <ale/ale_interface.hpp>
#include <cstdlib>
#include <filesystem>

class EnvironmentTest : public ::testing::Test {
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

  std::filesystem::path rom_path_;
};

// Tests that when truncation occurs, that the game over flag is set.
TEST_F(EnvironmentTest, TruncationWithMaxFramesPerEpisode) {
  // 485 is the number of frames required to reach a terminal state in Breakout,
  //  with this seed if you fired the ball and only ever called the fire action.
  const size_t max_frames_per_episode = 485;
  // Seed for reproducibility.
  const int seed = 42;
  // Shouldn't matter for this test.
  const bool grayscale = true;

  ai::environment::Environment env(rom_path_, max_frames_per_episode, grayscale,
                                   seed);

  // Reset the environment to start a new episode
  auto initial_obs = env.reset();
  EXPECT_FALSE(initial_obs.empty());

  ai::environment::Step step;
  for (size_t step_count = 0; step_count < max_frames_per_episode;
       ++step_count) {
    // Perform many no-op actions to eventually hit the truncation limit
    //  since the ball is never fired.
    step = env.step(ale::Action::PLAYER_A_NOOP);
    EXPECT_FALSE(step_count != max_frames_per_episode - 1 && step.truncated)
        << "Should not be truncated at step " << step_count + 1;
    // Since the ball is never fired, we should never hit a terminal state.
    EXPECT_FALSE(step.terminated) << "Game should not be terminated yet";
  }
  // After the loop, we should hit truncation
  EXPECT_TRUE(step.truncated)
      << "Expected truncation at " << max_frames_per_episode
      << " steps, but it was not hit.";
  EXPECT_FALSE(step.terminated) << "Should not be terminated when truncated";
  EXPECT_TRUE(step.game_over) << "Game should be over when truncated";
}

// Tests that when termination occurs, that the game over flag is set.
TEST_F(EnvironmentTest, TerminationFlagOnLossOfAllLives) {
  // Create environment with a very small max number of frames to trigger
  // truncation quickly
  const size_t max_frames_per_episode = 485;
  // Seed for reproducibility.
  const int seed = 42;
  // Shouldn't matter for this test.
  const bool grayscale = true;

  ai::environment::Environment env(rom_path_, max_frames_per_episode, grayscale,
                                   seed);

  // Reset the environment to start a new episode.
  auto initial_obs = env.reset();
  EXPECT_FALSE(initial_obs.empty());

  ai::environment::Step step;
  for (size_t step_count = 0; step_count < max_frames_per_episode;
       ++step_count) {
    // PLAYER_A_FIRE would start the game.
    step = env.step(ale::Action::PLAYER_A_FIRE);
    EXPECT_FALSE(step_count != max_frames_per_episode - 1 && step.terminated)
        << "Game ended naturally before truncation";
    EXPECT_FALSE(step.truncated)
        << "Should not be truncated at step " << step_count + 1;
  }
  EXPECT_TRUE(step.terminated);
  EXPECT_TRUE(step.game_over);
  EXPECT_FALSE(step.truncated);
}
