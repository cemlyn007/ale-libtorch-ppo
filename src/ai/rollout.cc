#include "rollout.h"

namespace ai::rollout {

Rollout::Rollout(std::filesystem::path rom_path, size_t horizon,
                 size_t num_episodes, size_t max_steps)
    : ale_(), rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface tmp_ale;
        tmp_ale.loadROM(rom_path);
        auto screen = tmp_ale.getScreen();
        return ai::buffer::Buffer(horizon, {screen.width(), screen.height()},
                                  1);
      }()),
      horizon_(horizon), num_episodes_(num_episodes), max_steps_(max_steps),
      is_terminal_(true), is_truncated_(true) {
  ale_.loadROM(rom_path_);
  observation_.resize(ale_.getScreen().width() * ale_.getScreen().height());
}

ale::Action Rollout::select_action() {
  return ale_.getMinimalActionSet()[rand() % ale_.getMinimalActionSet().size()];
}

void Rollout::get_observation() { ale_.getScreenGrayscale(observation_); }

void Rollout::rollout() {
  for (size_t i = 0; i < horizon_; i++) {
    if (is_terminal_ || is_truncated_) {
      buffer_.add(observation_, -1, -1, is_terminal_, is_truncated_);
      current_step_ = 0;
      is_terminal_ = false;
      is_truncated_ = false;
      ale_.reset_game();
      get_observation();
    } else {
      auto action = select_action();
      auto reward = ale_.act(action);
      buffer_.add(observation_, action, reward, is_terminal_, is_truncated_);
      get_observation();
      is_terminal_ = ale_.game_over();
      is_truncated_ = is_terminal_ ? false : current_step_ >= max_steps_;
      current_step_++;
    }
    if (is_terminal_ || is_truncated_) {
      current_episode_++;
    }
  }
}
} // namespace ai::rollout