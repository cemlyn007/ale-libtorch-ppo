#include <torch/torch.h>

namespace ai::buffer {

struct Batch {
  torch::Tensor observations;
  torch::Tensor actions;
  torch::Tensor rewards;
  torch::Tensor masks;
  torch::Tensor logits;
  torch::Tensor values;
  torch::Tensor advantages;
  torch::Tensor returns;
};

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

  Batch get(const torch::Tensor &next_values, float discount, float lambda);

private:
  torch::Device device_;
  size_t total_environments_;
  size_t capacity_;
  std::vector<int64_t> observation_shape_;
  int64_t indices_;

  torch::Tensor observations_;
  torch::Tensor actions_;
  torch::Tensor rewards_;
  torch::Tensor terminals_;
  torch::Tensor truncations_;
  torch::Tensor episode_starts_;
  torch::Tensor logits_;
  torch::Tensor values_;
  torch::Tensor advantages_;
  torch::Tensor returns_;
};

} // namespace ai::buffer