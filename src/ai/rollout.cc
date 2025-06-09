#include "rollout.h"

namespace ai::rollout {

Rollout::Rollout(std::filesystem::path rom_path, size_t horizon,
                 int num_episodes, int max_steps)
    : ale_(), rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface tmp_ale;
        tmp_ale.loadROM(rom_path);
        auto screen = tmp_ale.getScreen();
        return ai::buffer::Buffer(horizon, {screen.width(), screen.height()},
                                  1);
      }()),
      num_episodes_(num_episodes), max_steps_(max_steps) {
  ale_.loadROM(rom_path_);
}

ale::Action Rollout::select_action() {
  return ale_.getMinimalActionSet()[rand() % ale_.getMinimalActionSet().size()];
}

void Rollout::get_observation() { ale_.getScreenGrayscale(observation_); }

void Rollout::rollout() {
  while (current_episode_ < num_episodes_) {
    current_step_ = 0;
    is_terminal_ = false;
    is_truncated_ = false;
    ale_.reset_game();
    get_observation();
    while (!is_terminal_ && !is_truncated_) {
      auto action = select_action();
      auto reward = ale_.act(action);
      buffer_.add(observation_, action, reward, is_terminal_, is_truncated_);
      get_observation();
      is_terminal_ = ale_.game_over();
      is_truncated_ = is_terminal_ ? false : current_step_ >= max_steps_;
      current_step_++;
    }
    current_episode_++;
  }
}
} // namespace ai::rollout