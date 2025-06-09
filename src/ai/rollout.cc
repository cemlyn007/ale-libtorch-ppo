#include "rollout.h"
#include "algorithm"

namespace ai::rollout {

Rollout::Rollout(std::filesystem::path rom_path, size_t horizon,
                 size_t num_episodes, size_t max_steps, size_t frame_stack)
    : ale_(), rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface tmp_ale;
        tmp_ale.loadROM(rom_path);
        auto screen = tmp_ale.getScreen();
        return ai::buffer::Buffer(
            horizon, {frame_stack, screen.width(), screen.height()}, 1);
      }()),
      horizon_(horizon), num_episodes_(num_episodes), frame_stack_(frame_stack),
      max_steps_(max_steps), is_terminal_(true), is_truncated_(true) {
  ale_.loadROM(rom_path_);
  observation_.resize(frame_stack * ale_.getScreen().width() *
                      ale_.getScreen().height());
}

ale::Action Rollout::select_action() {
  return ale_.getMinimalActionSet()[rand() % ale_.getMinimalActionSet().size()];
}

void Rollout::get_reset_observation() {
  ale_.getScreenGrayscale(observation_);
  for (size_t i = 1; i < frame_stack_; i++)
    std::copy(observation_.begin(), observation_.end(),
              observation_.begin() +
                  i * ale_.getScreen().width() * ale_.getScreen().height());
}

void Rollout::get_observation() {
  std::shift_right(observation_.begin(), observation_.end(),
                   (frame_stack_ - 1) * ale_.getScreen().width() *
                       ale_.getScreen().height());
  ale_.getScreenGrayscale(observation_);
}

void Rollout::rollout() {
  for (size_t i = 0; i < horizon_; i++) {
    if (is_terminal_ || is_truncated_) {
      buffer_.add(observation_, -1, -1, is_terminal_, is_truncated_);
      current_step_ = 0;
      is_terminal_ = false;
      is_truncated_ = false;
      ale_.reset_game();
      get_reset_observation();
    } else {
      auto action = select_action();
      auto reward = ale_.act(action);
      buffer_.add(observation_, action, reward, is_terminal_, is_truncated_);
      get_observation();
      is_terminal_ = ale_.game_over(false);
      is_truncated_ =
          is_terminal_ ? false
                       : ale_.game_truncated() || current_step_ >= max_steps_;
      current_step_++;
    }
    if (is_terminal_ || is_truncated_) {
      current_episode_++;
    }
  }
}
} // namespace ai::rollout