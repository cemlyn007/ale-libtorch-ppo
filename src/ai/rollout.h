#include "ai/buffer.h"
#include "ai/environment/environment.h"
#include "ai/queue.h"
#include <atomic>
#include <filesystem>
#include <functional>
#include <torch/torch.h>

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
  void worker();

  std::filesystem::path rom_path_;
  size_t height_;
  size_t width_;
  ai::buffer::Buffer buffer_;
  std::vector<std::vector<unsigned char>> screen_buffers_;
  std::vector<torch::Tensor> screen_tensor_blobs_;
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
  std::vector<bool> is_episode_start_cpu_;
  std::vector<bool> game_overs_;
  std::vector<float> episode_returns_;
  std::vector<size_t> episode_lengths_;
  std::vector<float> game_returns_;
  std::vector<size_t> game_lengths_;
  torch::Tensor rewards_;
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

} // namespace ai::rollout