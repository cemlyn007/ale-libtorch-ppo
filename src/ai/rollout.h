#include "ale/ale_interface.hpp"
#include "buffer.h"
#include <filesystem>
#include <functional>
#include <memory>
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

class Rollout {
public:
  Rollout(std::filesystem::path rom_path, size_t total_environments,
          size_t horizon, size_t max_steps, size_t frame_stack,
          std::function<ActionResult(const torch::Tensor &)> action_selector,
          float gae_discount, float gae_lambda, const torch::Device &device,
          size_t seed);
  RolloutResult rollout();
  void get_observations();

  float gae_discount_ = 0.99f;
  float gae_lambda_ = 0.95f;

private:
  int64_t screen_width_;
  int64_t screen_height_;
  std::vector<std::unique_ptr<ale::ALEInterface>> ales_;
  std::string rom_path_;
  ai::buffer::Buffer buffer_;
  torch::Tensor observations_;
  int64_t total_environments_;
  size_t horizon_;
  int64_t frame_stack_;
  size_t max_steps_;
  size_t current_episode_ = 0;
  size_t total_steps_ = 0;
  float current_episode_return_ = 0.0f;
  size_t current_episode_length_ = 0;
  torch::Tensor is_terminal_;
  torch::Tensor is_truncated_;
  torch::Tensor is_episode_start_;
  std::vector<float> episode_returns_;
  std::vector<size_t> episode_lengths_;
  torch::Tensor rewards_;
  std::function<ActionResult(const torch::Tensor &)> action_selector_;
  torch::Device device_;
};

} // namespace ai::rollout