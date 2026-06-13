#include <torch/torch.h>

#include <ale/ale_interface.hpp>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <vector>

#include "ai/environment/environment.h"
#include "ai/environment/episode_life.h"
#include "ai/environment/fire_reset.h"
#include "ai/environment/max_and_skip.h"
#include "ai/environment/noop_reset.h"
#include "ai/environment/resize.h"
#include "ai/environment/truncate_on_episode_return.h"
#include "ai/rollout.h"
#include "gtest/gtest.h"

namespace {

using ai::environment::VirtualEnvironment;

// A representative Atari wrapper stack, exercising every stateful wrapper (noop
// RNG, max-and-skip frame buffer, episode-life lives, truncate accumulator)
// plus the ALE emulator underneath.
std::unique_ptr<VirtualEnvironment> make_stack(const std::filesystem::path &rom,
                                               size_t seed) {
  std::unique_ptr<VirtualEnvironment> env =
      std::make_unique<ai::environment::Environment>(rom, 108000,
                                                     /*grayscale=*/true,
                                                     static_cast<int>(seed));
  env = std::make_unique<ai::environment::NoopResetEnvironment>(std::move(env),
                                                                30, seed);
  env = std::make_unique<ai::environment::MaxAndSkipEnvironment>(std::move(env),
                                                                 4);
  env = std::make_unique<ai::environment::EpisodeLife>(std::move(env));
  env = std::make_unique<ai::environment::FireReset>(std::move(env));
  env = std::make_unique<ai::environment::ResizeEnvironment>(std::move(env), 84,
                                                             84);
  env = std::make_unique<ai::environment::TruncateOnEpisodeReturnEnvironment>(
      std::move(env), 864);
  return env;
}

class StateIoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *test_srcdir = std::getenv("TEST_SRCDIR");
    ASSERT_NE(test_srcdir, nullptr) << "TEST_SRCDIR is not set";
    rom_path_ =
        std::filesystem::path(test_srcdir) / "_main" / "roms" / "breakout.bin";
    ASSERT_TRUE(std::filesystem::exists(rom_path_))
        << "ROM not found: " << rom_path_;
  }

  std::filesystem::path rom_path_;
};

// Serializing a mid-episode env stack and restoring it into a fresh stack
// (seeded differently, so only the restored state can explain a match) must
// reproduce the exact same trajectory from the saved point on.
TEST_F(StateIoTest, EnvStackRoundTrips) {
  auto saved = make_stack(rom_path_, 7);
  saved->reset();
  for (const auto action :
       {ale::Action::PLAYER_A_NOOP, ale::Action::PLAYER_A_NOOP,
        ale::Action::PLAYER_A_RIGHT, ale::Action::PLAYER_A_LEFT})
    saved->step(action);

  std::stringstream blob;
  saved->serialize(blob);

  auto restored = make_stack(rom_path_, 99);  // different seed on purpose
  restored->deserialize(blob);

  for (const auto action :
       {ale::Action::PLAYER_A_NOOP, ale::Action::PLAYER_A_RIGHT,
        ale::Action::PLAYER_A_RIGHT, ale::Action::PLAYER_A_LEFT,
        ale::Action::PLAYER_A_NOOP, ale::Action::PLAYER_A_NOOP}) {
    const auto a = saved->step(action);
    const auto b = restored->step(action);
    EXPECT_EQ(a.reward, b.reward);
    EXPECT_EQ(a.terminated, b.terminated);
    EXPECT_EQ(a.truncated, b.truncated);
    EXPECT_EQ(a.game_over, b.game_over);
    EXPECT_EQ(a.observation, b.observation);
    if (a.game_over) break;
  }
}

// A Rollout saved to disk and restored into a fresh, identically-configured
// Rollout must produce a byte-identical next rollout (deterministic policy).
TEST_F(StateIoTest, RolloutStateRoundTripsOnDisk) {
  ale::ALEInterface ale;
  ale.loadROM(rom_path_.string());
  const auto action_size =
      static_cast<int64_t>(ale.getMinimalActionSet().size());

  const torch::Device device(torch::kCPU);
  const size_t num_environments = 2, horizon = 4;
  auto policy =
      [action_size](
          const torch::Tensor &observations) -> ai::rollout::ActionResult {
    const int64_t n = observations.size(0);
    return {torch::zeros({n}, torch::kLong),
            torch::zeros({n, action_size}, torch::kFloat32),
            torch::zeros({n}, torch::kFloat32)};
  };
  auto make_rollout = [&] {
    return std::make_unique<ai::rollout::Rollout>(
        rom_path_, num_environments, horizon, /*max_steps=*/108000,
        /*frame_stack=*/4, /*grayscale=*/true, policy, 0.99f, 0.95f, device,
        /*seed=*/0, /*num_workers=*/1, /*worker_batch_size=*/1,
        /*frame_skip=*/4, /*max_return=*/864.0f);
  };

  auto saved = make_rollout();
  saved->rollout();
  saved->rollout();
  const auto path =
      std::filesystem::temp_directory_path() / "ale_rollout_state_test.bin";
  saved->save_state(path);

  auto restored = make_rollout();
  restored->load_state(path);

  const auto from_saved = saved->rollout();
  const auto from_restored = restored->rollout();
  EXPECT_TRUE(
      from_saved.batch.observations.equal(from_restored.batch.observations))
      << "Restored rollout diverged from the saved one.";

  std::filesystem::remove(path);
}

}  // namespace
