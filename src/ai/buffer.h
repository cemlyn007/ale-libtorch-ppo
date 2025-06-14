#include <torch/torch.h>
#include <vector>

namespace ai::buffer {

class Buffer {
public:
  Buffer(size_t capacity, std::vector<size_t> observation_shape,
         size_t action_size);

  void add(const std::vector<unsigned char> &observation,
           const torch::Tensor &action, float reward, bool terminal,
           bool truncation, bool episode_start, const torch::Tensor &logits,
           const torch::Tensor &value);

  torch::Tensor observations_;
  torch::Tensor actions_;
  torch::Tensor rewards_;
  torch::Tensor terminals_;
  torch::Tensor truncations_;
  torch::Tensor episode_starts_;
  torch::Tensor logits_;
  torch::Tensor values_;

private:
  size_t capacity_;
  std::vector<int64_t> observation_shape_;
  size_t indices_;
};

} // namespace ai::buffer