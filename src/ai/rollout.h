#include <torch/torch.h>

#include <atomic>
#include <filesystem>
#include <functional>

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
  Rollout(std::filesystem::path rom_path, size_t total_environments,
          size_t horizon, size_t max_steps, size_t frame_stack, bool grayscale,
          std::function<ActionResult(const torch::Tensor &)> action_selector,
          float gae_discount, float gae_lambda, const torch::Device &device,
          size_t seed, size_t num_workers, size_t worker_batch_size,
          size_t frame_skip, ale::reward_t max_return = 0.0f,
          std::optional<std::filesystem::path> video_path = std::nullopt,
          bool record_observation = false);
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
  std::vector<StepResult> step_all();
  void upload_step_state();
  void worker();

  std::filesystem::path rom_path_;
  size_t height_;
  size_t width_;
  ai::buffer::Buffer buffer_;
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

  std::atomic<bool> stop_;

  std::vector<std::thread> workers_;
  ai::queue::Queue<size_t> action_queue_;
  ai::queue::Queue<StepResult> step_queue_;
  size_t batch_size_;
  bool grayscale_;
  bool record_observation_;

  ActionResult action_result_;
};

}  // namespace ai::rollout