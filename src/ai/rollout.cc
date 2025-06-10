#include "rollout.h"
#include "algorithm"
#include "gae.h"

namespace ai::rollout {

Rollout::Rollout(
    std::filesystem::path rom_path, size_t horizon, size_t num_episodes,
    size_t max_steps, size_t frame_stack,
    std::function<ActionResult(const torch::Tensor &)> action_selector)
    : ale_(), rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface ale;
        ale.loadROM(rom_path);
        auto screen = ale.getScreen();
        return ai::buffer::Buffer(
            horizon, {frame_stack, screen.width(), screen.height()},
            ale.getMinimalActionSet().size());
      }()),
      horizon_(horizon), num_episodes_(num_episodes), frame_stack_(frame_stack),
      max_steps_(max_steps), is_terminal_(true), is_truncated_(true),
      action_selector_(action_selector) {
  ale_.loadROM(rom_path_);
  observation_.resize(frame_stack * ale_.getScreen().width() *
                      ale_.getScreen().height());
}

ActionResult Rollout::select_action() {
  auto w = static_cast<int64_t>(ale_.getScreen().width());
  auto h = static_cast<int64_t>(ale_.getScreen().height());
  auto fs = static_cast<int64_t>(frame_stack_);
  auto obs_tensor =
      torch::from_blob(observation_.data(), {fs, w, h}, torch::kByte).clone();
  return action_selector_(obs_tensor);
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

Batch Rollout::rollout() {
  ActionResult action_result = select_action();
  for (size_t i = 0; i < horizon_; i++) {
    if (is_terminal_ || is_truncated_) {
      buffer_.add(observation_, -1, -1, is_terminal_, is_truncated_,
                  is_episode_start_, action_result.logits, action_result.value);
      current_step_ = 0;
      is_terminal_ = false;
      is_truncated_ = false;
      is_episode_start_ = false;
      ale_.reset_game();
      get_reset_observation();
    } else {
      action_result = select_action();
      auto reward = ale_.act(action_result.action);
      buffer_.add(observation_, action_result.action, reward, is_terminal_,
                  is_episode_start_, is_truncated_, action_result.logits,
                  action_result.value);
      get_observation();
      is_terminal_ = ale_.game_over(false);
      is_truncated_ =
          is_terminal_ ? false
                       : ale_.game_truncated() || current_step_ >= max_steps_;
      current_step_++;
    }
    if (is_terminal_ || is_truncated_) {
      is_episode_start_ = true;
      current_episode_++;
    }
  }
  action_result = select_action();
  auto advantages =
      ai::gae::gae(buffer_.rewards_, buffer_.values_, action_result.value,
                   buffer_.terminals_, buffer_.truncations_,
                   buffer_.episode_starts_, 0.99, 0.95);
  auto returns = advantages + buffer_.values_;
  return Batch{buffer_.observations_,
               buffer_.actions_,
               buffer_.rewards_,
               torch::logical_not(buffer_.episode_starts_),
               buffer_.values_,
               advantages,
               returns};
}
} // namespace ai::rollout