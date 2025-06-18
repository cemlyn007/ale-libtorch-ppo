#include "rollout.h"
#include "gae.h"
#include <cassert>

namespace ai::rollout {

Rollout::Rollout(
    std::filesystem::path rom_path, size_t total_environments, size_t horizon,
    size_t max_steps, size_t frame_stack,
    std::function<ActionResult(const torch::Tensor &)> action_selector,
    float gae_discount, float gae_lambda, const torch::Device &device)
    : gae_discount_(gae_discount), gae_lambda_(gae_lambda), ales_(),
      rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface ale;
        ale.loadROM(rom_path);
        auto screen = ale.getScreen();
        return ai::buffer::Buffer(
            total_environments, horizon,
            {frame_stack, screen.width(), screen.height()},
            ale.getMinimalActionSet().size(), device);
      }()),
      total_environments_(total_environments), horizon_(horizon),
      frame_stack_(frame_stack), max_steps_(max_steps), is_terminal_(),
      is_truncated_(), is_episode_start_(), action_selector_(action_selector),
      device_(device) {

  // TODO: Initialize vectors etc
  if (total_environments_ == 0) {
    throw std::invalid_argument("Total environments must be greater than 0.");
  }
  if (horizon_ == 0) {
    throw std::invalid_argument("Horizon must be greater than 0.");
  }
  if (max_steps_ == 0) {
    throw std::invalid_argument("Max steps must be greater than 0.");
  }
  if (frame_stack_ == 0) {
    throw std::invalid_argument("Frame stack must be greater than 0.");
  }
  if (rom_path_.empty()) {
    throw std::invalid_argument("ROM path must not be empty.");
  }
  if (!std::filesystem::exists(rom_path_)) {
    throw std::invalid_argument("ROM file does not exist: " + rom_path_);
  }

  for (int64_t i = 0; i < total_environments_; i++) {
    ales_.push_back(std::make_unique<ale::ALEInterface>());
    ales_.back()->setBool("truncate_on_loss_of_life", true);
    ales_.back()->setInt("max_num_frames_per_episode", 108000);
    ales_.back()->setInt("frame_skip", 1);
    ales_.back()->setInt("random_seed", i);
    ales_.back()->setFloat("repeat_action_probability", 0.0f);
    ales_.back()->loadROM(rom_path_);
    assert(ales_.back()->getInt("max_num_frames_per_episode") == 108000);
    screen_width_ = ales_.back()->getScreen().width();
    screen_height_ = ales_.back()->getScreen().height();
  }

  observations_ = torch::zeros(
      {total_environments_, frame_stack_, screen_width_, screen_height_},
      torch::TensorOptions(torch::kByte).device(device_));
  is_terminal_ =
      torch::zeros({total_environments_},
                   torch::TensorOptions(torch::kBool).device(device_));
  is_truncated_ =
      torch::zeros({total_environments_},
                   torch::TensorOptions(torch::kBool).device(device_));
  is_episode_start_ =
      torch::ones({total_environments_},
                  torch::TensorOptions(torch::kBool).device(device_));
  rewards_ =
      torch::zeros({total_environments_},
                   torch::TensorOptions(torch::kFloat32).device(device_));
  episode_returns_.resize(total_environments_, 0.0f);
  episode_lengths_.resize(total_environments_, 0);
}

void Rollout::get_observations() {
  for (int64_t frame_index = frame_stack_ - 1; frame_index > 0; --frame_index) {
    observations_.index_put_(
        {torch::indexing::Slice(), frame_index},
        observations_.index({torch::indexing::Slice(), frame_index - 1}));
  }
  std::vector<unsigned char> gray_scale(screen_width_ * screen_height_);
  for (int64_t environment_index = 0; environment_index < total_environments_;
       ++environment_index) {
    ales_[environment_index]->getScreenGrayscale(gray_scale);
    auto frame = torch::from_blob(
        gray_scale.data(), {screen_width_, screen_height_}, torch::kByte);
    if (is_episode_start_[environment_index].item<bool>()) {
      for (int64_t frame_index = 0; frame_index < frame_stack_; ++frame_index) {
        observations_.index_put_({environment_index, frame_index}, frame);
      }
    } else {
      observations_.index_put_({environment_index, 0}, frame);
    }
  }
}

RolloutResult Rollout::rollout() {
  get_observations();
  ActionResult action_result = action_selector_(observations_);

  std::vector<float> episode_returns;
  std::vector<size_t> episode_lengths;

  for (size_t time_index = 0; time_index < horizon_; time_index++) {
    for (int64_t ale_index = 0; ale_index < total_environments_; ++ale_index) {
      if (is_episode_start_[ale_index].item<bool>()) {
        ales_[ale_index]->reset_game();
      } else {
        auto ale_action_set = ales_[ale_index]->getMinimalActionSet();
        int64_t action_index = action_result.actions[ale_index].item<int64_t>();
        if (action_index < 0 ||
            action_index >= static_cast<int64_t>(ale_action_set.size())) {
          throw std::out_of_range("Action index out of range for environment " +
                                  std::to_string(ale_index));
        }
        auto ale_action = ale_action_set[action_index];
        auto reward = ales_[ale_index]->act(ale_action);
        bool terminal = ales_[ale_index]->game_over(false);
        bool truncated = (!terminal) && ales_[ale_index]->game_truncated();

        rewards_[ale_index] = reward;
        is_terminal_[ale_index] = terminal;
        is_truncated_[ale_index] = truncated;

        episode_returns_[ale_index] += reward;
        episode_lengths_[ale_index]++;
      }
    }

    // Add the observations, and the actions that from those observations led to
    // the rewards and terminal state changes.
    buffer_.add(observations_, action_result.actions, rewards_, is_terminal_,
                is_truncated_, is_episode_start_, action_result.logits,
                action_result.values);

    // Get the next observations after taking actions.
    get_observations();

    for (int64_t ale_index = 0; ale_index < total_environments_; ++ale_index) {
      if (is_terminal_[ale_index].item<bool>() ||
          is_truncated_[ale_index].item<bool>()) {
        is_episode_start_[ale_index] = true;
        is_terminal_[ale_index] = false;
        is_truncated_[ale_index] = false;
        current_episode_++;
        episode_returns.push_back(episode_returns_[ale_index]);
        episode_lengths.push_back(episode_lengths_[ale_index]);
        episode_returns_[ale_index] = 0.0;
        episode_lengths_[ale_index] = 0;
      } else {
        is_episode_start_[ale_index] = false;
      }
    }

    // Select actions for the next step.
    action_result = action_selector_(observations_);
    total_steps_ += total_environments_;
  }

  auto batch = buffer_.get(action_result.values, gae_discount_, gae_lambda_);

  Log log{total_steps_, current_episode_, episode_returns, episode_lengths};

  return {batch, log};
}
} // namespace ai::rollout