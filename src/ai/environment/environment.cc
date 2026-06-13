#include "ai/environment/environment.h"

#include "ai/environment/state_io.h"

namespace ai::environment {

Environment::Environment(const std::filesystem::path &rom_path,
                         size_t max_num_frames_per_episode, bool grayscale,
                         int seed)
    : ale_(), grayscale_(grayscale), size_(0) {
  if (rom_path.empty())
    throw std::invalid_argument("ROM path must not be empty.");
  if (!std::filesystem::exists(rom_path))
    throw std::invalid_argument("ROM file does not exist: " +
                                rom_path.string());
  ale_.setInt("max_num_frames_per_episode",
              static_cast<int>(max_num_frames_per_episode));
  ale_.setInt("frame_skip", 1);
  ale_.setFloat("repeat_action_probability", 0.0f);
  ale_.setInt("random_seed", seed);
  // Settings only take effect on load, so the screen size (ROM-dependent)
  // must be read after this single load.
  ale_.loadROM(rom_path.string());
  const auto &screen = ale_.getScreen();
  size_ = (grayscale_ ? 1 : 3) * screen.height() * screen.width();
}

ScreenBuffer Environment::reset() {
  ale_.reset_game();
  return get_observation();
}

Step Environment::step(const ale::Action &action, bool want_observation) {
  ale::reward_t reward = ale_.act(action);
  bool terminated = ale_.game_over(false);
  bool truncated = ale_.game_truncated() && !terminated;
  // If we have exceeded our max frames, we consider the game to be over as
  // well.
  bool game_over = terminated || truncated;
  return {.observation = want_observation ? get_observation() : ScreenBuffer(),
          .reward = reward,
          .terminated = terminated,
          .truncated = truncated,
          .game_over = game_over};
}

ale::ALEInterface &Environment::get_interface() { return ale_; }

void Environment::serialize(std::ostream &os) {
  // cloneSystemState() captures the emulator state including the RNG.
  ale::ALEState state = ale_.cloneSystemState();
  state_io::write_bytes(os, state.serialize());
}

void Environment::deserialize(std::istream &is) {
  ale::ALEState state(state_io::read_bytes(is));
  ale_.restoreState(state);
}

ScreenBuffer Environment::get_observation() {
  ScreenBuffer observation(size_);
  if (grayscale_)
    ale_.getScreenGrayscale(observation);
  else
    ale_.getScreenRGB(observation);
  return observation;
}

}  // namespace ai::environment
