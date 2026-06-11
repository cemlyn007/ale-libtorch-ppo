#include "ai/environment/environment.h"

#include <ale/ale_interface.hpp>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ai/environment/max_and_skip.h"
#include "gtest/gtest.h"

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

namespace {

// Deterministic stand-in for MaxAndSkip tests: frame n is the single pixel
// {n}, and grabbed frame numbers are recorded, so pooling and grab-skipping
// are observable from the outside.
class FakeEnvironment : public ai::environment::VirtualEnvironment {
 public:
  explicit FakeEnvironment(int terminate_at_frame = 0)
      : terminate_at_frame_(terminate_at_frame) {}
  using VirtualEnvironment::step;
  ai::environment::ScreenBuffer reset() override {
    frame_ = 0;
    return {static_cast<unsigned char>(frame_)};
  }
  ai::environment::Step step(const ale::Action &,
                             bool want_observation) override {
    ++frame_;
    if (want_observation) grabbed_frames.push_back(frame_);
    const bool terminated =
        terminate_at_frame_ != 0 && frame_ >= terminate_at_frame_;
    ai::environment::ScreenBuffer observation;
    if (want_observation)
      observation.push_back(static_cast<unsigned char>(frame_));
    return {.observation = std::move(observation),
            .reward = 1,
            .terminated = terminated,
            .truncated = false,
            .game_over = terminated};
  }
  ale::ALEInterface &get_interface() override {
    throw std::logic_error("FakeEnvironment has no ALE interface.");
  }

  std::vector<int> grabbed_frames;

 private:
  int terminate_at_frame_;
  int frame_ = 0;
};

}  // namespace

// The emitted observation pools the last two frames of the skip window, the
// earlier frames' screens are never grabbed, and rewards sum over the window.
TEST(MaxAndSkipEnvironmentTest, PoolsLastTwoFramesAndSkipsEarlierGrabs) {
  auto fake = std::make_unique<FakeEnvironment>();
  auto *fake_ptr = fake.get();
  ai::environment::MaxAndSkipEnvironment env(std::move(fake), 4);

  env.reset();
  auto step = env.step(ale::Action::PLAYER_A_NOOP);
  EXPECT_EQ(step.observation, ai::environment::ScreenBuffer{4});
  EXPECT_EQ(step.reward, 4);
  EXPECT_EQ(fake_ptr->grabbed_frames, (std::vector<int>{3, 4}));

  step = env.step(ale::Action::PLAYER_A_NOOP);
  EXPECT_EQ(step.observation, ai::environment::ScreenBuffer{8});
  EXPECT_EQ(fake_ptr->grabbed_frames, (std::vector<int>{3, 4, 7, 8}));
}

// Termination once the pooling window has been entered emits the frame that
// was grabbed on the terminal step.
TEST(MaxAndSkipEnvironmentTest, TerminationInPoolingWindowEmitsTerminalFrame) {
  auto fake = std::make_unique<FakeEnvironment>(/*terminate_at_frame=*/3);
  ai::environment::MaxAndSkipEnvironment env(std::move(fake), 4);

  env.reset();
  auto step = env.step(ale::Action::PLAYER_A_NOOP);
  EXPECT_TRUE(step.terminated);
  EXPECT_EQ(step.reward, 3);
  EXPECT_EQ(step.observation, ai::environment::ScreenBuffer{3});
}

// Termination before the pooling window means no frame was grabbed; the
// terminal step must still emit a correctly-sized observation (the most
// recent real frame, here the reset frame).
TEST(MaxAndSkipEnvironmentTest, EarlyTerminationFallsBackToLastRealFrame) {
  auto fake = std::make_unique<FakeEnvironment>(/*terminate_at_frame=*/2);
  auto *fake_ptr = fake.get();
  ai::environment::MaxAndSkipEnvironment env(std::move(fake), 4);

  env.reset();
  auto step = env.step(ale::Action::PLAYER_A_NOOP);
  EXPECT_TRUE(step.terminated);
  EXPECT_EQ(step.reward, 2);
  EXPECT_TRUE(fake_ptr->grabbed_frames.empty());
  EXPECT_EQ(step.observation, ai::environment::ScreenBuffer{0});
}
