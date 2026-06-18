#include "rollout.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <numeric>

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
    std::optional<std::filesystem::path> video_path, bool record_observation,
    size_t pipeline_groups, size_t num_buffers)
    : gae_discount_(gae_discount),
      gae_lambda_(gae_lambda),
      rom_path_(rom_path),
      height_(84),
      width_(84),
      buffers_([&] {
        ale::ALEInterface ale;
        ale.loadROM(rom_path);
        std::vector<size_t> observation_shape;
        if (grayscale)
          observation_shape = {frame_stack, height_, width_};
        else
          observation_shape = {frame_stack, 3, height_, width_};
        const size_t action_size = ale.getMinimalActionSet().size();
        std::vector<ai::buffer::Buffer> buffers;
        buffers.reserve(num_buffers);
        for (size_t i = 0; i < num_buffers; ++i)
          buffers.emplace_back(total_environments, horizon, observation_shape,
                               action_size, device);
        return buffers;
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
      record_observation_(record_observation),
      num_groups_(pipeline_groups),
      group_size_(pipeline_groups == 0 ? 0
                                       : total_environments / pipeline_groups) {
  if (total_environments_ == 0) {
    throw std::invalid_argument("Total environments must be greater than 0.");
  }
  if (horizon_ == 0) {
    throw std::invalid_argument("Horizon must be greater than 0.");
  }
  if (num_groups_ == 0) {
    throw std::invalid_argument("Pipeline groups must be greater than 0.");
  }
  if (buffers_.empty()) {
    throw std::invalid_argument("Number of buffers must be greater than 0.");
  }
  if (total_environments_ % num_groups_ != 0) {
    throw std::invalid_argument(
        "Pipeline groups must divide total environments.");
  }
  if (batch_size_ == 0) {
    throw std::invalid_argument("Worker batch size must be greater than 0.");
  }
  // Workers pop fixed batches of batch_size_ from the action queue; a
  // remainder sub-batch never satisfies any worker's pop predicate, so
  // rollout() would deadlock waiting for the missing results.
  if (total_environments_ % batch_size_ != 0) {
    throw std::invalid_argument("Total environments (" +
                                std::to_string(total_environments_) +
                                ") must be divisible by worker batch size (" +
                                std::to_string(batch_size_) + ").");
  }
  // A group is dispatched as one batch of work units; a remainder smaller
  // than worker_batch_size would leave Queue::pop(batch) starved forever.
  if (group_size_ % worker_batch_size != 0) {
    throw std::invalid_argument(
        "Worker batch size must divide the pipeline group size.");
  }
  if (max_steps_ == 0) {
    throw std::invalid_argument("Max steps must be greater than 0.");
  }
  if (frame_stack_ == 0) {
    throw std::invalid_argument("Frame stack must be greater than 0.");
  }
  // With zero workers nothing services action_queue_, so the first rollout()
  // would block forever.
  if (num_workers == 0) {
    throw std::invalid_argument("Number of workers must be greater than 0.");
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
      // Pin to cores 1..N-1, never core 0, so the unpinned main thread always
      // has a core no pinned thread can occupy. (N can be reported as 0.)
      const unsigned n = std::thread::hardware_concurrency();
      if (n > 1) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(1 + (i % (n - 1)), &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
      }
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

  // Workers index this by absolute env id each step; preallocating lets each
  // group stage its slice with one D2H copy while other groups keep stepping.
  actions_cpu_ = torch::empty({static_cast<int64_t>(total_environments_)},
                              torch::TensorOptions(torch::kLong));
  group_results_.resize(num_groups_);
  for (size_t g = 0; g < num_groups_; ++g) {
    step_queues_.emplace_back(std::make_unique<ai::queue::Queue<StepResult>>());
    auto &indices = group_indices_.emplace_back(group_size_);
    std::iota(indices.begin(), indices.end(), g * group_size_);
  }

  spdlog::info("Creating {} worker threads.", num_workers);
  // Workers stay unpinned: a pinned worker sharing the main thread's core
  // would starve it, and ffmpeg (popen'd from a worker on episode reset)
  // inherits the caller's mask. Configs size num_workers to cores-1 instead.
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

  // Probe the ROM so the Atari-specific wrappers below only apply where they
  // fit: EpisodeLife throws on a 0-life game, and FIRE-to-start needs FIRE.
  ale::ALEInterface &interface = environment->get_interface();
  const bool has_lives = interface.lives() > 0;
  const auto minimal_actions = interface.getMinimalActionSet();
  const bool has_fire =
      std::find(minimal_actions.begin(), minimal_actions.end(),
                ale::Action::PLAYER_A_FIRE) != minimal_actions.end();

  // Some games cap out at a known max score (e.g. Breakout); truncate there.
  // max_return <= 0 disables it (a 0 ceiling would throw on the first step).
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
  // One observation per skip window, so real-time playback is 60/frame_skip.
  if (i == 0 && video_path.has_value() && record_observation_)
    environment = std::make_unique<ai::environment::EpisodeObservationRecorder>(
        std::move(environment), video_path.value(), grayscale_ ? 1 : 3, height_,
        width_, 60 / frame_skip);
  // Only games that track lives end an episode on life loss; EpisodeLife
  // throws on a 0-life game (e.g. Pong), so skip it there.
  if (has_lives)
    environment =
        std::make_unique<ai::environment::EpisodeLife>(std::move(environment));
  // FIRE-to-start (Gym's FireResetEnv) only applies when FIRE is an action.
  if (has_fire)
    environment =
        std::make_unique<ai::environment::FireReset>(std::move(environment));
  return environment;
}

Rollout::~Rollout() {
  // Wake the workers blocked in action_queue_.pop() so they observe stop_ and
  // exit. The batch contents don't matter: worker() skips the work once stop_
  // is set, so teardown never steps or resets envs that are being destroyed --
  // a reset() of env 0 here would re-open its still-open video recorder.
  stop_ = true;
  std::vector<size_t> inputs(total_environments_);
  std::iota(inputs.begin(), inputs.end(), size_t{0});
  action_queue_.push(inputs);
  for (auto &worker : workers_)
    if (worker.joinable()) worker.join();
}

void Rollout::update_observations() {
  update_observations(0, total_environments_);
}

void Rollout::update_observations(size_t env_start, size_t env_count) {
  const auto start = static_cast<int64_t>(env_start);
  const auto count = static_cast<int64_t>(env_count);
  auto observations = observations_.narrow(0, start, count);
  // Shift the stack back by one; the oldest frame drops off.
  for (int64_t frame_index = frame_stack_ - 1; frame_index > 0; --frame_index)
    observations.index_put_(
        {torch::indexing::Slice(), frame_index},
        observations.index({torch::indexing::Slice(), frame_index - 1}));
  // On a fresh episode, prime every stacked frame with the first observation so
  // the stack isn't polluted by the previous episode's tail.
  for (size_t i = env_start; i < env_start + env_count; ++i)
    if (is_episode_start_cpu_[i])
      observations_.select(0, i).copy_(staging_.select(0, i),
                                       /*non_blocking=*/true);
  // Newest frame for these envs in a single page-locked H2D upload.
  observations.select(1, 0).copy_(staging_.narrow(0, start, count),
                                  /*non_blocking=*/true);
}

void Rollout::upload_step_state(size_t env_start, size_t env_count) {
  // Wrap the persistent host mirrors (no copy) and push each to its accelerator
  // tensor in one go. copy_ handles the dtype cast (byte mirror -> bool
  // tensor).
  const auto start = static_cast<int64_t>(env_start);
  const auto n = static_cast<int64_t>(env_count);
  const auto byte = torch::TensorOptions(torch::kByte);
  rewards_.narrow(0, start, n)
      .copy_(torch::from_blob(rewards_host_.data() + env_start, {n},
                              torch::kFloat32));
  is_terminated_.narrow(0, start, n)
      .copy_(torch::from_blob(terminated_host_.data() + env_start, {n}, byte));
  is_truncated_.narrow(0, start, n)
      .copy_(torch::from_blob(truncated_host_.data() + env_start, {n}, byte));
  is_episode_start_.narrow(0, start, n)
      .copy_(torch::from_blob(is_episode_start_cpu_.data() + env_start, {n},
                              byte));
}

void Rollout::pump_group(size_t group) {
  const auto start = static_cast<int64_t>(group * group_size_);
  const auto count = static_cast<int64_t>(group_size_);
  // Action selection for this group only. Pull the actions to the host once
  // (one sync) so workers index them without a per-env device->host copy.
  group_results_[group] =
      action_selector_(observations_.narrow(0, start, count));
  actions_cpu_.narrow(0, start, count).copy_(group_results_[group].actions);
  // Hand the group to the workers; they step while the main thread services
  // the other groups (post-processing and inference) -- the pipeline overlap.
  action_queue_.push(group_indices_[group]);
}

void Rollout::post_group(size_t group, size_t time_index, Log &log) {
  const size_t env_start = group * group_size_;
  const auto start = static_cast<int64_t>(env_start);
  const auto count = static_cast<int64_t>(group_size_);
  const auto step_results = step_queues_[group]->pop(group_size_);

  size_t total_steps_increment = 0;
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

  // Upload this step's per-env state in bulk copies instead of O(envs)
  // synchronising scalar writes.
  upload_step_state(env_start, group_size_);

  // Add the observations, and the actions that from those observations led
  // to the rewards and terminal state changes.
  active_buffer().add_rows(
      start, count, static_cast<int64_t>(time_index),
      observations_.narrow(0, start, count), group_results_[group].actions,
      rewards_.narrow(0, start, count), is_terminated_.narrow(0, start, count),
      is_truncated_.narrow(0, start, count),
      is_episode_start_.narrow(0, start, count), group_results_[group].logits,
      group_results_[group].values);

  // Get the next observations after taking actions and saving the
  // observations.
  update_observations(env_start, group_size_);

  for (const auto &result : step_results) {
    int64_t ale_index = result.environment_index;
    if (result.terminated || result.truncated) {
      // Accelerator state, via host mirrors (uploaded next step):
      is_episode_start_cpu_[ale_index] = 1;
      terminated_host_[ale_index] = 0;
      truncated_host_[ale_index] = 0;
      // On the CPU:
      current_episode_++;
      log.episode_returns.push_back(episode_returns_[ale_index]);
      log.episode_lengths.push_back(episode_lengths_[ale_index]);
      episode_returns_[ale_index] = 0.0;
      episode_lengths_[ale_index] = 0;
      if (game_overs_[ale_index]) {
        log.game_returns.push_back(game_returns_[ale_index]);
        log.game_lengths.push_back(game_lengths_[ale_index]);
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

RolloutResult Rollout::rollout() {
  Log log{};

  // Software pipeline over env groups in a fixed round-robin (fixed order
  // keeps RNG consumption deterministic): every group is dispatched before
  // any is drained, so workers always have another group's steps queued
  // while the main thread runs this group's post-processing and forward.
  // With one group this degenerates to the classic forward->step->record
  // loop, op for op.
  for (size_t group = 0; group < num_groups_; ++group) pump_group(group);
  for (size_t time_index = 0; time_index < horizon_; time_index++) {
    for (size_t group = 0; group < num_groups_; ++group) {
      post_group(group, time_index, log);
      if (time_index + 1 < horizon_) pump_group(group);
    }
  }
  // Bootstrap values for GAE from the observations after the final step.
  const auto action_result = action_selector_(observations_);
  const auto batch =
      active_buffer().get(action_result.values, gae_discount_, gae_lambda_);
  // Rotate so the next rollout fills a different buffer and the batch just
  // returned stays valid while the caller consumes it.
  active_buffer_index_ = (active_buffer_index_ + 1) % buffers_.size();
  log.steps = total_steps_;
  log.episodes = current_episode_;
  return {batch, log};
}

void Rollout::worker() {
  while (!stop_) {
    auto inputs = action_queue_.pop(batch_size_);
    for (const auto &input : inputs) {
      // ~Rollout() pushes a final batch only to unblock the pop() above; once
      // stop_ is set there is no live rollout reading results, so skip the work
      // rather than step/reset (and re-open env 0's recorder) on dying envs.
      if (stop_) break;
      StepResult result = step(input);
      step_queues_[input / group_size_]->push(result);
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