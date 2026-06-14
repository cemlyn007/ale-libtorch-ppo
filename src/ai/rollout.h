#pragma once
#include <torch/torch.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>

#include "ai/buffer.h"
#include "ai/environment/environment.h"
#include "ai/queue.h"

namespace ai::rollout {

struct Log {
  size_t steps;
  size_t episodes;
  std::vector<float> episode_returns;
  std::vector<size_t> episode_lengths;
  std::vector<float> game_returns;
  std::vector<size_t> game_lengths;
};

struct RolloutResult {
  ai::buffer::Batch batch;
  Log log;
};

struct ActionResult {
  torch::Tensor actions;
  torch::Tensor logits;
  torch::Tensor values;
};

struct StepResult {
  size_t environment_index;
  float reward;
  bool terminated;
  bool truncated;
  bool game_over;
};

class Rollout {
 public:
  // num_buffers: rollout buffers cycled round-robin across rollout() calls.
  // The returned batch aliases the buffer it was collected into, so 2 lets a
  // caller train on rollout k while rollout k+1 is collected (double
  // buffering); 1 reuses the same storage every call.
  Rollout(std::filesystem::path rom_path, size_t total_environments,
          size_t horizon, size_t max_steps, size_t frame_stack, bool grayscale,
          std::function<ActionResult(const torch::Tensor &)> action_selector,
          float gae_discount, float gae_lambda, const torch::Device &device,
          size_t seed, size_t num_workers, size_t worker_batch_size,
          size_t frame_skip, ale::reward_t max_return = 0.0f,
          std::optional<std::filesystem::path> video_path = std::nullopt,
          bool record_observation = false, size_t pipeline_groups = 1,
          size_t num_buffers = 1);
  ~Rollout();
  RolloutResult rollout();
  void update_observations();

  float gae_discount_ = 0.99f;
  float gae_lambda_ = 0.95f;

 private:
  std::unique_ptr<ai::environment::VirtualEnvironment> create_environment(
      size_t i, size_t seed, size_t frame_skip, ale::reward_t max_return,
      const std::optional<std::filesystem::path> &video_path) const;
  StepResult step(const size_t environment_index);
  // Pipelined rollout stages, operating on one contiguous env group each.
  // pump: policy forward + stage actions on the host + dispatch to workers.
  // post: drain the group's results, account into log, upload, record,
  // advance obs.
  void pump_group(size_t group);
  void post_group(size_t group, size_t time_index, Log &log);
  void upload_step_state(size_t env_start, size_t env_count);
  void update_observations(size_t env_start, size_t env_count);
  void worker();

  ai::buffer::Buffer &active_buffer() { return buffers_[active_buffer_index_]; }

  std::filesystem::path rom_path_;
  size_t height_;
  size_t width_;
  // Rotated by one after every rollout() (see num_buffers in the constructor).
  std::vector<ai::buffer::Buffer> buffers_;
  size_t active_buffer_index_ = 0;
  // Single page-locked staging buffer [total_environments, ...frame_shape] for
  // the newest frame of every env. Workers memcpy into disjoint slices; one
  // async H2D copy per step feeds the frame stack.
  torch::Tensor staging_;
  uint8_t *staging_ptr_ = nullptr;
  int64_t frame_bytes_ = 0;
  torch::Tensor observations_;
  size_t total_environments_;
  size_t horizon_;
  size_t frame_stack_;
  size_t max_steps_;
  size_t current_episode_ = 0;
  size_t total_steps_ = 0;
  torch::Tensor is_terminated_;
  torch::Tensor is_truncated_;
  torch::Tensor is_episode_start_;
  // Persistent host mirrors of the four per-env accelerator tensors above.
  // Mutated on the CPU each step, then uploaded in one bulk copy each,
  // replacing O(envs) synchronising scalar writes. is_episode_start_cpu_
  // doubles as the CPU-side gate; uint8_t (not bool) so it is contiguous for
  // from_blob.
  std::vector<float> rewards_host_;
  std::vector<uint8_t> terminated_host_;
  std::vector<uint8_t> truncated_host_;
  std::vector<uint8_t> is_episode_start_cpu_;
  std::vector<bool> game_overs_;
  std::vector<float> episode_returns_;
  std::vector<size_t> episode_lengths_;
  std::vector<float> game_returns_;
  std::vector<size_t> game_lengths_;
  torch::Tensor rewards_;
  // Host copy of the latest selected actions, refreshed once per step so
  // workers index it without a per-env device->host sync.
  torch::Tensor actions_cpu_;
  std::function<ActionResult(const torch::Tensor &)> action_selector_;
  torch::Device device_;

  std::vector<std::unique_ptr<ai::environment::VirtualEnvironment>>
      environments_;
  // Same ROM in every env, so one copy serves all workers (read-only after
  // construction). getMinimalActionSet() builds a fresh vector per call, so
  // it must not be called per step.
  std::vector<ale::Action> minimal_action_set_;

  std::atomic<bool> stop_;

  std::vector<std::thread> workers_;
  ai::queue::Queue<size_t> action_queue_;
  // One result queue per pipeline group so draining a group is a single
  // blocking pop and never mixes with another group's in-flight results.
  // (Queue holds a mutex, so the queues are heap-allocated to stay put.)
  std::vector<std::unique_ptr<ai::queue::Queue<StepResult>>> step_queues_;
  size_t batch_size_;
  bool grayscale_;
  bool record_observation_;

  // Pipelined inference: envs split into num_groups_ contiguous groups of
  // group_size_; while one group steps in the workers, the main thread
  // post-processes and re-dispatches the others. 1 group == the classic
  // fully-synchronous loop.
  size_t num_groups_;
  size_t group_size_;
  std::vector<std::vector<size_t>> group_indices_;  // constant dispatch lists
  std::vector<ActionResult> group_results_;         // in-flight per group
};

}  // namespace ai::rollout