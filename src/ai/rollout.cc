#include "rollout.h"
#include "algorithm"
#include "gae.h"

namespace ai::rollout {

Rollout::Rollout(
    std::filesystem::path rom_path, size_t horizon, size_t max_steps,
    size_t frame_stack,
    std::function<ActionResult(const torch::Tensor &)> action_selector,
    float gae_gamma, float gae_lambda)
    : gae_gamma_(gae_gamma), gae_lambda_(gae_lambda), ale_(),
      rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface ale;
        ale.loadROM(rom_path);
        auto screen = ale.getScreen();
        return ai::buffer::Buffer(
            horizon, {frame_stack, screen.width(), screen.height()},
            ale.getMinimalActionSet().size());
      }()),
      horizon_(horizon), frame_stack_(frame_stack), max_steps_(max_steps),
      is_terminal_(true), is_truncated_(true), is_episode_start_(true),
      action_selector_(action_selector) {
  ale_.loadROM(rom_path_);
  ale_.setBool("truncate_on_loss_of_life", true);
  observation_.resize(frame_stack * ale_.getScreen().width() *
                      ale_.getScreen().height());
}

ActionResult Rollout::select_action() {
  auto w = static_cast<int64_t>(ale_.getScreen().width());
  auto h = static_cast<int64_t>(ale_.getScreen().height());
  auto fs = static_cast<int64_t>(frame_stack_);
  auto observation =
      torch::from_blob(observation_.data(), {fs, w, h}, torch::kByte).clone();
  return action_selector_(observation);
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

RolloutResult Rollout::rollout() {
  ActionResult action_result = select_action();
  std::vector<float> episode_returns;
  std::vector<size_t> episode_lengths;

  for (size_t i = 0; i < horizon_; i++) {
    if (is_episode_start_) {
      buffer_.add(observation_, 0, 0, is_terminal_, is_truncated_,
                  is_episode_start_, action_result.logits, action_result.value);
      current_step_ = 0;
      is_episode_start_ = false;
      ale_.reset_game();
      get_reset_observation();
    } else {
      action_result = select_action();
      auto reward = ale_.act(action_result.action);
      is_terminal_ = ale_.game_over(false);
      is_truncated_ =
          is_terminal_ ? false
                       : ale_.game_truncated() || current_step_ >= max_steps_;

      current_episode_return_ += static_cast<float>(reward);
      current_episode_length_++;
      total_steps_++;

      buffer_.add(observation_, action_result.action, reward, is_terminal_,
                  is_episode_start_, is_truncated_, action_result.logits,
                  action_result.value);
      get_observation();
      current_step_++;
    }
    if (is_terminal_ || is_truncated_) {
      is_episode_start_ = true;
      is_terminal_ = false;
      is_truncated_ = false;
      current_episode_++;
      episode_returns.push_back(current_episode_return_);
      episode_lengths.push_back(current_episode_length_);
      current_episode_return_ = 0.0f;
      current_episode_length_ = 0;
    }
  }

  action_result = select_action();
  auto advantages =
      ai::gae::gae(buffer_.rewards_, buffer_.values_, action_result.value,
                   buffer_.terminals_, buffer_.truncations_,
                   buffer_.episode_starts_, gae_gamma_, gae_lambda_);
  auto returns = advantages + buffer_.values_;

  Batch batch{buffer_.observations_,
              buffer_.actions_,
              buffer_.rewards_,
              torch::logical_not(buffer_.episode_starts_),
              buffer_.logits_,
              buffer_.values_,
              advantages,
              returns};

  Log log{total_steps_, current_episode_, episode_returns, episode_lengths};

  return {batch, log};
}
} // namespace ai::rollout