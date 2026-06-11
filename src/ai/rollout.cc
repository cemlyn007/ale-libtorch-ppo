#include "rollout.h"

#include <spdlog/spdlog.h>

#include <cassert>

#include "ai/environment/episode_life.h"
#include "ai/environment/episode_observation_recorder.h"
#include "ai/environment/episode_recorder.h"
#include "ai/environment/fire_reset.h"
#include "ai/environment/max_and_skip.h"
#include "ai/environment/noop_reset.h"
#include "ai/environment/resize.h"
#include "ai/environment/truncate_on_episode_return.h"
#include "ai/gae.h"

namespace ai::rollout {

Rollout::Rollout(
    std::filesystem::path rom_path, size_t total_environments, size_t horizon,
    size_t max_steps, size_t frame_stack, bool grayscale,
    std::function<ActionResult(const torch::Tensor &)> action_selector,
    float gae_discount, float gae_lambda, const torch::Device &device,
    size_t seed, size_t num_workers, size_t worker_batch_size,
    size_t frame_skip, ale::reward_t max_return,
    std::optional<std::filesystem::path> video_path, bool record_observation)
    : gae_discount_(gae_discount),
      gae_lambda_(gae_lambda),
      rom_path_(rom_path),
      height_(84),
      width_(84),
      buffer_([&] {
        ale::ALEInterface ale;
        ale.loadROM(rom_path);
        std::vector<size_t> observation_shape;
        if (grayscale)
          observation_shape = {frame_stack, height_, width_};
        else
          observation_shape = {frame_stack, 3, height_, width_};
        return ai::buffer::Buffer(total_environments, horizon,
                                  observation_shape,
                                  ale.getMinimalActionSet().size(), device);
      }()),
      total_environments_(total_environments),
      horizon_(horizon),
      frame_stack_(frame_stack),
      max_steps_(max_steps),
      is_terminated_(),
      is_truncated_(),
      is_episode_start_(),
      game_overs_(total_environments, false),
      episode_returns_(total_environments, 0.0f),
      episode_lengths_(total_environments, 0),
      game_returns_(total_environments, 0.0f),
      game_lengths_(total_environments, 0),
      action_selector_(action_selector),
      device_(device),
      environments_(),
      stop_(),
      batch_size_(worker_batch_size),
      grayscale_(grayscale),
      record_observation_(record_observation) {
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

  // Per-frame observation shape (channels-last for grayscale, CHW for colour)
  // and its byte count -- the unit each worker writes into the staging buffer.
  std::vector<int64_t> frame_shape =
      grayscale_ ? std::vector<int64_t>{static_cast<int64_t>(height_),
                                        static_cast<int64_t>(width_)}
                 : std::vector<int64_t>{3, static_cast<int64_t>(height_),
                                        static_cast<int64_t>(width_)};
  frame_bytes_ = grayscale_ ? static_cast<int64_t>(height_ * width_)
                            : static_cast<int64_t>(3 * height_ * width_);

  // One contiguous, genuinely page-locked staging buffer for every env's newest
  // frame. (from_blob over a std::vector does NOT pin -- pinned_memory only
  // takes effect on allocating factories like empty(), and only with CUDA.)
  std::vector<int64_t> staging_shape = {
      static_cast<int64_t>(total_environments_)};
  staging_shape.insert(staging_shape.end(), frame_shape.begin(),
                       frame_shape.end());
  auto staging_options = torch::TensorOptions(torch::kByte);
  if (device_.is_cuda()) staging_options = staging_options.pinned_memory(true);
  staging_ = torch::empty(staging_shape, staging_options);
  staging_ptr_ = staging_.data_ptr<uint8_t>();

  environments_.resize(total_environments_);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < total_environments_; ++i) {
    threads.emplace_back([&, i]() {
#if defined(__linux__)
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
      environments_[i] =
          create_environment(i, seed, frame_skip, max_return, video_path);
    });
  }
  for (auto &thread : threads) thread.join();

  minimal_action_set_ = environments_[0]->get_interface().getMinimalActionSet();

  auto total = static_cast<int64_t>(total_environments_);
  auto frame = static_cast<int64_t>(frame_stack_);
  auto options = torch::TensorOptions(torch::kFloat32).device(device_);
  std::vector<int64_t> observations_shape({total, frame});
  observations_shape.insert(observations_shape.end(), frame_shape.begin(),
                            frame_shape.end());

  observations_ = torch::zeros(observations_shape, options.dtype(torch::kByte));
  is_terminated_ = torch::zeros({total}, options.dtype(torch::kBool));
  is_truncated_ = torch::zeros({total}, options.dtype(torch::kBool));
  is_episode_start_ = torch::ones({total}, options.dtype(torch::kBool));
  rewards_ = torch::zeros({total}, options);

  // Host mirrors start matching the accelerator tensors above.
  rewards_host_.resize(total_environments_, 0.0f);
  terminated_host_.resize(total_environments_, 0);
  truncated_host_.resize(total_environments_, 0);
  is_episode_start_cpu_.resize(total_environments_, 1);

  spdlog::info("Creating {} worker threads.", num_workers);
  for (size_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back(&Rollout::worker, this);
  }
}

std::unique_ptr<ai::environment::VirtualEnvironment>
Rollout::create_environment(
    size_t i, size_t seed, size_t frame_skip, ale::reward_t max_return,
    const std::optional<std::filesystem::path> &video_path) const {
  std::unique_ptr<ai::environment::VirtualEnvironment> environment =
      std::make_unique<ai::environment::Environment>(rom_path_, max_steps_,
                                                     grayscale_, i + seed);

  // Atari breakout only has two sets of bricks, once the second set is
  // cleared, no more bricks will appear.
  if (max_return > 0.0f)
    environment =
        std::make_unique<ai::environment::TruncateOnEpisodeReturnEnvironment>(
            std::move(environment), max_return);

  // The full-screen recorder grabs RGB straight off the interface, so it stays
  // below MaxAndSkip to capture every emulator frame (60fps video).
  if (i == 0 && video_path.has_value() && !record_observation_)
    environment = std::make_unique<ai::environment::EpisodeRecorder>(
        std::move(environment), video_path.value(), false);

  // TODO: Make this configurable.
  environment = std::make_unique<ai::environment::NoopResetEnvironment>(
      std::move(environment), 30, seed + i);
  environment = std::make_unique<ai::environment::MaxAndSkipEnvironment>(
      std::move(environment), frame_skip);
  // Resize sits above MaxAndSkip so flicker pooling happens at native
  // resolution and only the one emitted frame per skip window pays the resize.
  environment = std::make_unique<ai::environment::ResizeEnvironment>(
      std::move(environment), width_, height_);
  // Above Resize so it records exactly what the agent sees (pooled + resized).
  if (i == 0 && video_path.has_value() && record_observation_)
    environment = std::make_unique<ai::environment::EpisodeObservationRecorder>(
        std::move(environment), video_path.value(), grayscale_ ? 1 : 3, height_,
        width_);
  environment =
      std::make_unique<ai::environment::EpisodeLife>(std::move(environment));
  environment =
      std::make_unique<ai::environment::FireReset>(std::move(environment));
  return environment;
}

Rollout::~Rollout() {
  stop_ = true;
  std::vector<size_t> inputs(total_environments_);
  action_result_.actions.fill_(ale::Action::RANDOM);
  is_episode_start_cpu_.assign(total_environments_, true);
  for (size_t i = 0; i < total_environments_; ++i) inputs[i] = i;
  action_queue_.push(inputs);
  for (auto &worker : workers_)
    if (worker.joinable()) worker.join();
}

void Rollout::update_observations() {
  // Shift the stack back by one; the oldest frame drops off.
  for (int64_t frame_index = frame_stack_ - 1; frame_index > 0; --frame_index)
    observations_.index_put_(
        {torch::indexing::Slice(), frame_index},
        observations_.index({torch::indexing::Slice(), frame_index - 1}));
  // On a fresh episode, prime every stacked frame with the first observation so
  // the stack isn't polluted by the previous episode's tail.
  for (size_t i = 0; i < total_environments_; ++i)
    if (is_episode_start_cpu_[i])
      observations_.select(0, i).copy_(staging_.select(0, i),
                                       /*non_blocking=*/true);
  // Newest frame for every env in a single page-locked H2D upload.
  observations_.select(1, 0).copy_(staging_, /*non_blocking=*/true);
}

void Rollout::upload_step_state() {
  // Wrap the persistent host mirrors (no copy) and push each to its accelerator
  // tensor in one go. copy_ handles the dtype cast (byte mirror -> bool
  // tensor).
  const int64_t n = static_cast<int64_t>(total_environments_);
  const auto byte = torch::TensorOptions(torch::kByte);
  rewards_.copy_(torch::from_blob(rewards_host_.data(), {n}, torch::kFloat32));
  is_terminated_.copy_(torch::from_blob(terminated_host_.data(), {n}, byte));
  is_truncated_.copy_(torch::from_blob(truncated_host_.data(), {n}, byte));
  is_episode_start_.copy_(
      torch::from_blob(is_episode_start_cpu_.data(), {n}, byte));
}

RolloutResult Rollout::rollout() {
  std::vector<float> episode_returns;
  std::vector<size_t> episode_lengths;
  std::vector<float> game_returns;
  std::vector<size_t> game_lengths;

  for (size_t time_index = 0; time_index < horizon_; time_index++) {
    // Action Selection. Pull the actions to the host once (one sync) so workers
    // index them without a per-env device->host copy. kLong + contiguous so
    // workers can read through a raw pointer.
    action_result_ = action_selector_(observations_);
    actions_cpu_ =
        action_result_.actions.to(torch::kCPU, torch::kInt64).contiguous();

    // Step all environments with the selected actions.
    size_t total_steps_increment = 0;
    const auto step_results = step_all();
    for (const auto &result : step_results) {
      int64_t ale_index = result.environment_index;
      if (!is_episode_start_cpu_[ale_index]) {
        // Host mirror of the accelerator tensors (uploaded in bulk below):
        rewards_host_[ale_index] = result.reward;
        terminated_host_[ale_index] = result.terminated;
        truncated_host_[ale_index] = result.truncated;
        // On the CPU:
        game_overs_[ale_index] = result.game_over;
        episode_returns_[ale_index] += result.reward;
        episode_lengths_[ale_index]++;
        game_returns_[ale_index] += result.reward;
        game_lengths_[ale_index]++;
        total_steps_increment++;
      }
    }

    // Upload this step's per-env state in four bulk copies instead of O(envs)
    // synchronising scalar writes.
    upload_step_state();

    // Add the observations, and the actions that from those observations led
    // to the rewards and terminal state changes.
    buffer_.add(observations_, action_result_.actions, rewards_, is_terminated_,
                is_truncated_, is_episode_start_, action_result_.logits,
                action_result_.values);

    // Get the next observations after taking actions and saving the
    // observations.
    update_observations();

    for (const auto &result : step_results) {
      int64_t ale_index = result.environment_index;
      if (result.terminated || result.truncated) {
        // Accelerator state, via host mirrors (uploaded next step):
        is_episode_start_cpu_[ale_index] = 1;
        terminated_host_[ale_index] = 0;
        truncated_host_[ale_index] = 0;
        // On the CPU:
        current_episode_++;
        episode_returns.push_back(episode_returns_[ale_index]);
        episode_lengths.push_back(episode_lengths_[ale_index]);
        episode_returns_[ale_index] = 0.0;
        episode_lengths_[ale_index] = 0;
        if (game_overs_[ale_index]) {
          game_returns.push_back(game_returns_[ale_index]);
          game_lengths.push_back(game_lengths_[ale_index]);
          game_returns_[ale_index] = 0.0;
          game_lengths_[ale_index] = 0;
        }
      } else if (is_episode_start_cpu_[ale_index]) {
        // Accelerator state, via host mirror (uploaded next step):
        is_episode_start_cpu_[ale_index] = 0;
      }
    }
    total_steps_ += total_steps_increment;
  }
  action_result_ = action_selector_(observations_);
  const auto batch =
      buffer_.get(action_result_.values, gae_discount_, gae_lambda_);
  const Log log{.steps = total_steps_,
                .episodes = current_episode_,
                .episode_returns = episode_returns,
                .episode_lengths = episode_lengths,
                .game_returns = game_returns,
                .game_lengths = game_lengths};
  return {batch, log};
}

std::vector<StepResult> Rollout::step_all() {
  std::vector<size_t> inputs(total_environments_);
  for (size_t i = 0; i < total_environments_; ++i) {
    inputs[i] = i;
  }
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

StepResult Rollout::step(const size_t environment_index) {
  StepResult output;
  output.environment_index = environment_index;
  std::vector<unsigned char> observation;
  if (is_episode_start_cpu_[environment_index]) {
    observation = environments_[environment_index]->reset();
    output.reward = 0.0f;
    output.terminated = false;
    output.truncated = false;
    output.game_over = false;
  } else {
    const size_t action_index =
        actions_cpu_.const_data_ptr<int64_t>()[environment_index];
    if (action_index >= minimal_action_set_.size())
      throw std::out_of_range("Action index out of range for environment " +
                              std::to_string(environment_index));
    auto action = minimal_action_set_[action_index];
    auto result = environments_[environment_index]->step(action);
    observation = std::move(result.observation);
    output.reward = result.reward;
    output.terminated = result.terminated;
    output.truncated = result.truncated;
    output.game_over = result.game_over;
  }
  std::memcpy(staging_ptr_ + environment_index * frame_bytes_,
              observation.data(), observation.size());
  return output;
}
}  // namespace ai::rollout