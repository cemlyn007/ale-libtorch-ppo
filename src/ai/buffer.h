#include <torch/torch.h>
#include <vector>

namespace ai::buffer {

class Buffer {
public:
  Buffer(size_t capacity, std::vector<size_t> observation_shape,
         size_t action_size);

  void add(std::vector<unsigned char> observation, int action, float reward,
           bool terminal, bool truncation, torch::Tensor logits,
           torch::Tensor value);

  torch::Tensor observations_;
  torch::Tensor actions_;
  torch::Tensor rewards_;
  torch::Tensor terminals_;
  torch::Tensor truncations_;
  torch::Tensor logits_;
  torch::Tensor values_;

private:
  size_t capacity_;
  std::vector<int64_t> observation_shape_;
  size_t indices_;
};

} // namespace ai::buffer