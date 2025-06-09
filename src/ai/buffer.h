#include <torch/torch.h>
#include <vector>

namespace ai::buffer {

class Buffer {
public:
  Buffer(int capacity, std::vector<size_t> observation_shape, int action_size);

  void add(std::vector<unsigned char> observation, int action, float reward,
           bool terminal, bool truncation);

private:
  int capacity_;
  std::vector<int64_t> observation_shape_;
  torch::Tensor observations_;
  torch::Tensor actions_;
  torch::Tensor rewards_;
  torch::Tensor terminals_;
  torch::Tensor truncations_;

  int indices_;
};

} // namespace ai::buffer