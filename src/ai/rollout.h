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

struct StepInput {
  size_t environment_index;
  ale::Action action;
  bool is_episode_start;
};

struct StepResult {
  size_t environment_index;
  float reward;
  bool terminated;
  bool truncated;
};

class Rollout {
public:
  Rollout(std::filesystem::path rom_path, size_t total_environments,
          size_t horizon, size_t max_steps, size_t frame_stack,
          std::function<ActionResult(const torch::Tensor &)> action_selector,
          float gae_discount, float gae_lambda, const torch::Device &device,
          size_t seed, size_t num_workers, size_t worker_batch_size,
          size_t frame_skip,
          std::optional<std::filesystem::path> video_path = std::nullopt);
  ~Rollout();
  RolloutResult rollout();
  void update_observations();

  float gae_discount_ = 0.99f;
  float gae_lambda_ = 0.95f;

private:
  StepResult step(const StepInput &);
  std::vector<StepResult> step_all(const std::vector<StepInput> &inputs);
  void worker();

  std::filesystem::path rom_path_;
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
  float current_episode_return_ = 0.0f;
  size_t current_episode_length_ = 0;
  torch::Tensor is_terminated_;
  torch::Tensor is_truncated_;
  torch::Tensor is_episode_start_;
  std::vector<float> episode_returns_;
  std::vector<size_t> episode_lengths_;
  torch::Tensor rewards_;
  std::function<ActionResult(const torch::Tensor &)> action_selector_;
  torch::Device device_;

  std::vector<std::unique_ptr<ai::environment::VirtualEnvironment>>
      environments_;

  std::atomic<bool> stop_;

  std::vector<std::thread> workers_;
  ai::queue::Queue<StepInput> action_queue_;
  ai::queue::Queue<StepResult> step_queue_;
  size_t batch_size_;
};

} // namespace ai::rollout