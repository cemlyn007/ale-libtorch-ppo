#include <torch/torch.h>

namespace ai::buffer {

class Buffer {
public:
  Buffer(size_t total_environments, size_t capacity,
         std::vector<size_t> observation_shape, size_t action_size,
         const torch::Device &device);

  void add(const torch::Tensor &observations, const torch::Tensor &actions,
           const torch::Tensor &rewards, const torch::Tensor &terminals,
           const torch::Tensor &truncations,
           const torch::Tensor &episode_starts, const torch::Tensor &logits,
           const torch::Tensor &values);

  torch::Tensor observations_;
  torch::Tensor actions_;
  torch::Tensor rewards_;
  torch::Tensor terminals_;
  torch::Tensor truncations_;
  torch::Tensor episode_starts_;
  torch::Tensor logits_;
  torch::Tensor values_;

private:
  torch::Device device_;
  size_t total_environments_;
  size_t capacity_;
  std::vector<int64_t> observation_shape_;
  int64_t indices_;
};

} // namespace ai::buffer