#include "ai/environment/environment.h"

namespace ai::environment {

Environment::Environment(const std::filesystem::path &rom_path,
                         size_t max_num_frames_per_episode, bool grayscale,
                         int seed)
    : ale_(), grayscale_(grayscale), size_([&] {
        ale_.loadROM(rom_path.string());
        auto screen = ale_.getScreen();
        auto channels = grayscale_ ? 1 : 3;
        return channels * screen.height() * screen.width();
      }()) {
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
  ale_.loadROM(rom_path.string());
}

ScreenBuffer Environment::reset() {
  ale_.reset_game();
  return get_observation();
}

Step Environment::step(const ale::Action &action) {
  ale::reward_t reward = ale_.act(action);
  bool game_over = ale_.game_over(false);
  return {.observation = get_observation(),
          .reward = reward,
          .terminated = game_over,
          .truncated = ale_.game_truncated(),
          .game_over = game_over};
}

ale::ALEInterface &Environment::get_interface() { return ale_; }

ScreenBuffer Environment::get_observation() {
  ScreenBuffer observation(size_);
  if (grayscale_)
    ale_.getScreenGrayscale(observation);
  else
    ale_.getScreenRGB(observation);
  return observation;
}

} // namespace ai::environment
