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
    size_t frame_skip, std::optional<std::filesystem::path> video_path)
    : gae_discount_(gae_discount), gae_lambda_(gae_lambda), rom_path_(rom_path),
      buffer_([&] {
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
      device_(device), environments_(), stop_(),
      batch_size_(worker_batch_size) {
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
    throw std::invalid_argument(std::string("ROM file does not exist: ") +
                                rom_path_.string());
  }

  for (size_t i = 0; i < total_environments_; i++) {
    auto environment = std::make_unique<ai::environment::Environment>(
        rom_path_, max_steps_, frame_skip, 0.0f, i + seed);
    std::unique_ptr<ai::environment::EpisodeLife> episode_life;
    if (i == 0 && video_path.has_value()) {
      auto recorder = std::make_unique<ai::environment::EpisodeRecorder>(
          std::move(environment), video_path.value());
      episode_life =
          std::make_unique<ai::environment::EpisodeLife>(std::move(recorder));
    } else {
      episode_life = std::make_unique<ai::environment::EpisodeLife>(
          std::move(environment));
    }
    environments_.emplace_back(std::move(episode_life));
    auto screen = environments_.back()->get_interface().getScreen();
    screen_buffers_.emplace_back(
        std::vector<unsigned char>(screen.height() * screen.width()));
    environments_.back()->get_interface().getScreenGrayscale(
        screen_buffers_.back());
    screen_tensor_blobs_.push_back(
        torch::from_blob(screen_buffers_.back().data(),
                         {static_cast<int64_t>(screen.height()),
                          static_cast<int64_t>(screen.width())},
                         torch::kByte));
  }

  auto total = static_cast<int64_t>(total_environments_);
  auto frame = static_cast<int64_t>(frame_stack_);
  auto options = torch::TensorOptions(torch::kFloat32).device(device_);
  observations_ =
      torch::zeros({total, frame, screen_tensor_blobs_.back().size(0),
                    screen_tensor_blobs_.back().size(1)},
                   options.dtype(torch::kByte));
  is_terminated_ = torch::zeros({total}, options.dtype(torch::kBool));
  is_truncated_ = torch::zeros({total}, options.dtype(torch::kBool));
  is_episode_start_ = torch::ones({total}, options.dtype(torch::kBool));
  rewards_ = torch::zeros({total}, options);

  episode_returns_.resize(total_environments_, 0.0f);
  std::fill(episode_returns_.begin(), episode_returns_.end(), 0.0);
  episode_lengths_.resize(total_environments_, 0);
  std::fill(episode_lengths_.begin(), episode_lengths_.end(), 0);

  std::cout << "Creating " << num_workers << " worker threads." << std::endl;
  for (size_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back(&Rollout::worker, this);
  }
}

Rollout::~Rollout() {
  stop_ = true;
  std::vector<StepInput> inputs(total_environments_);
  for (size_t i = 0; i < total_environments_; ++i)
    inputs[i] = StepInput{i, ale::Action::RANDOM, true};
  action_queue_.push(inputs);
  for (auto &worker : workers_)
    if (worker.joinable())
      worker.join();
}

void Rollout::update_observations() {
  for (int64_t frame_index = frame_stack_ - 1; frame_index > 0; --frame_index)
    observations_.index_put_(
        {torch::indexing::Slice(), frame_index},
        observations_.index({torch::indexing::Slice(), frame_index - 1}));
  for (size_t i = 0; i < total_environments_; ++i) {
    const auto &frame = screen_tensor_blobs_[i];
    if (is_episode_start_[i].item<bool>()) {
      observations_.select(0, i).copy_(frame);
    } else {
      observations_.index_put_({static_cast<int64_t>(i), 0}, frame);
    }
  }
}

RolloutResult Rollout::rollout() {
  std::vector<float> episode_returns;
  std::vector<size_t> episode_lengths;

  ActionResult action_result;
  std::vector<StepInput> step_inputs(total_environments_);
  for (size_t time_index = 0; time_index < horizon_; time_index++) {

    // Action Selection
    action_result = action_selector_(observations_);
    const auto actions = action_result.actions;
    const auto is_episode_start = is_episode_start_.to(torch::kCPU);
    for (size_t ale_index = 0; ale_index < total_environments_; ++ale_index) {
      auto &interface = environments_[ale_index]->get_interface();
      auto action_set = interface.getMinimalActionSet();
      size_t action_index = actions[ale_index].item<int64_t>();
      if (action_index < 0 || action_index >= action_set.size())
        throw std::out_of_range("Action index out of range for environment " +
                                std::to_string(ale_index));
      auto action = action_set[action_index];
      bool episode_start = is_episode_start[ale_index].item<bool>();
      step_inputs[ale_index] = StepInput{ale_index, action, episode_start};
    }

    // Step all environments with the selected actions.
    size_t total_steps_increment = 0;
    const auto step_results = step_all(step_inputs);
    for (const auto &result : step_results) {
      int64_t ale_index = result.environment_index;
      if (!step_inputs[ale_index].is_episode_start) {
        rewards_[ale_index] = result.reward;
        is_terminated_[ale_index] = result.terminated;
        is_truncated_[ale_index] = result.truncated;
        episode_returns_[ale_index] += result.reward;
        episode_lengths_[ale_index]++;
        total_steps_increment++;
      }
    }

    // Add the observations, and the actions that from those observations led to
    // the rewards and terminal state changes.
    buffer_.add(observations_, action_result.actions, rewards_, is_terminated_,
                is_truncated_, is_episode_start_, action_result.logits,
                action_result.values);

    // Get the next observations after taking actions and saving the
    // observations.
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
    total_steps_ += total_steps_increment;
  }
  action_result = action_selector_(observations_);
  const auto batch =
      buffer_.get(action_result.values, gae_discount_, gae_lambda_);
  const Log log{total_steps_, current_episode_, episode_returns,
                episode_lengths};
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
    environments_[input.environment_index]->reset();
    output.reward = 0.0f;
    output.terminated = false;
    output.truncated = false;
  } else {
    auto result = environments_[input.environment_index]->step(input.action);
    output.reward = result.reward;
    output.terminated = result.terminated;
    output.truncated = result.truncated;
  }
  environments_[input.environment_index]->get_interface().getScreenGrayscale(
      screen_buffers_[input.environment_index]);
  return output;
}
} // namespace ai::rollout