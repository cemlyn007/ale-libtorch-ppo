#include "rollout.h"
#include "gae.h"
#include <cassert>

namespace ai::rollout {

Rollout::Rollout(
    std::filesystem::path rom_path, size_t total_environments, size_t horizon,
    size_t max_steps, size_t frame_stack,
    std::function<ActionResult(const torch::Tensor &)> action_selector,
    float gae_discount, float gae_lambda, const torch::Device &device,
    size_t seed, size_t num_workers, size_t worker_batch_size,
    size_t frame_skip)
    : gae_discount_(gae_discount), gae_lambda_(gae_lambda), ales_(),
      rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface ale;
        ale.loadROM(rom_path);
        auto screen = ale.getScreen();
        return ai::buffer::Buffer(
            total_environments, horizon,
            {frame_stack, screen.height(), screen.width()},
            ale.getMinimalActionSet().size(), device);
      }()),
      total_environments_(total_environments), horizon_(horizon),
      frame_stack_(frame_stack), max_steps_(max_steps), is_terminated_(),
      is_truncated_(), is_episode_start_(), action_selector_(action_selector),
      device_(device), stop_(), batch_size_(worker_batch_size) {
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

  for (size_t i = 0; i < total_environments_; i++) {
    ales_.push_back(std::make_unique<ale::ALEInterface>());
    ales_.back()->setBool("truncate_on_loss_of_life", true);
    ales_.back()->setInt("max_num_frames_per_episode", max_steps_);
    ales_.back()->setInt("frame_skip", static_cast<int>(frame_skip));
    ales_.back()->setInt("random_seed", i + seed);
    ales_.back()->setFloat("repeat_action_probability", 0.0f);
    ales_.back()->loadROM(rom_path_);
    assert(ales_.back()->getInt("max_num_frames_per_episode") ==
           static_cast<int>(max_steps_));
    screen_width_ = ales_.back()->getScreen().width();
    screen_height_ = ales_.back()->getScreen().height();
  }

  auto total = static_cast<int64_t>(total_environments_);
  auto frame = static_cast<int64_t>(frame_stack_);
  observations_ =
      torch::zeros({total, frame, screen_height_, screen_width_},
                   torch::TensorOptions(torch::kByte).device(device_));
  is_terminated_ =
      torch::zeros({total}, torch::TensorOptions(torch::kBool).device(device_));
  is_truncated_ =
      torch::zeros({total}, torch::TensorOptions(torch::kBool).device(device_));
  is_episode_start_ =
      torch::ones({total}, torch::TensorOptions(torch::kBool).device(device_));
  rewards_ = torch::zeros(
      {total}, torch::TensorOptions(torch::kFloat32).device(device_));
  episode_returns_.resize(total_environments_, 0.0f);
  episode_lengths_.resize(total_environments_, 0);

  std::cout << "Creating " << num_workers << " worker threads." << std::endl;
  for (size_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back(&Rollout::worker, this);
  }
}

Rollout::~Rollout() {
  stop_ = true;
  for (size_t i = 0; i < total_environments_; ++i)
    action_queue_.push(std::vector<StepInput>{});
  for (auto &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void Rollout::update_observations() {
  for (int64_t frame_index = frame_stack_ - 1; frame_index > 0; --frame_index) {
    observations_.index_put_(
        {torch::indexing::Slice(), frame_index},
        observations_.index({torch::indexing::Slice(), frame_index - 1}));
  }
  std::vector<unsigned char> gray_scale(screen_height_ * screen_width_);
  auto frame = torch::from_blob(gray_scale.data(),
                                {screen_height_, screen_width_}, torch::kByte);
  for (size_t i = 0; i < total_environments_; ++i) {
    ales_[i]->getScreenGrayscale(gray_scale);
    if (is_episode_start_[i].item<bool>()) {
      observations_.select(0, i).copy_(frame);
    } else {
      observations_.index_put_({static_cast<int64_t>(i), 0}, frame);
    }
  }
}

RolloutResult Rollout::rollout() {
  update_observations();
  ActionResult action_result = action_selector_(observations_);

  std::vector<float> episode_returns;
  std::vector<size_t> episode_lengths;

  std::vector<StepInput> step_inputs(total_environments_);
  for (size_t time_index = 0; time_index < horizon_; time_index++) {
    for (size_t ale_index = 0; ale_index < total_environments_; ++ale_index) {
      auto ale_action_set = ales_[ale_index]->getMinimalActionSet();
      int64_t action_index =
          action_result.actions[static_cast<int64_t>(ale_index)]
              .item<int64_t>();
      if (action_index < 0 ||
          action_index >= static_cast<int64_t>(ale_action_set.size())) {
        throw std::out_of_range("Action index out of range for environment " +
                                std::to_string(ale_index));
      }
      auto action = ale_action_set[action_index];
      step_inputs[ale_index] = StepInput{
          ale_index, action, is_episode_start_[ale_index].item<bool>()};
    }

    auto step_results = step_all(step_inputs);

    for (const auto &result : step_results) {
      int64_t ale_index = result.environment_index;
      if (!step_inputs[ale_index].is_episode_start) {
        rewards_[ale_index] = result.reward;
        is_terminated_[ale_index] = result.terminated;
        is_truncated_[ale_index] = result.truncated;
        episode_returns_[ale_index] += result.reward;
        episode_lengths_[ale_index]++;
      }
    }

    // Add the observations, and the actions that from those observations led to
    // the rewards and terminal state changes.
    buffer_.add(observations_, action_result.actions, rewards_, is_terminated_,
                is_truncated_, is_episode_start_, action_result.logits,
                action_result.values);

    // Get the next observations after taking actions.
    update_observations();

    for (const auto &result : step_results) {
      int64_t ale_index = result.environment_index;
      if (result.terminated || result.truncated) {
        is_episode_start_[ale_index] = true;
        is_terminated_[ale_index] = false;
        is_truncated_[ale_index] = false;
        current_episode_++;
        episode_returns.push_back(episode_returns_[ale_index]);
        episode_lengths.push_back(episode_lengths_[ale_index]);
        episode_returns_[ale_index] = 0.0;
        episode_lengths_[ale_index] = 0;
      } else if (step_inputs[ale_index].is_episode_start) {
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

std::vector<StepResult>
Rollout::step_all(const std::vector<StepInput> &inputs) {
  action_queue_.push(inputs);
  return step_queue_.pop(inputs.size());
}

void Rollout::worker() {
  while (!stop_) {
    auto inputs = action_queue_.pop(batch_size_);
    for (const auto &input : inputs) {
      StepResult result = step(input);
      step_queue_.push(result);
    }
  }
}

StepResult Rollout::step(const StepInput &input) {
  StepResult output;
  output.environment_index = input.environment_index;
  if (input.is_episode_start) {
    ales_[input.environment_index]->reset_game();
    output.reward = 0.0f;
    output.terminated = false;
    output.truncated = false;
  } else {
    output.reward = ales_[input.environment_index]->act(input.action);
    output.terminated = ales_[input.environment_index]->game_over(false);
    output.truncated =
        (!output.terminated) & ales_[input.environment_index]->game_truncated();
  }
  return output;
}
} // namespace ai::rollout