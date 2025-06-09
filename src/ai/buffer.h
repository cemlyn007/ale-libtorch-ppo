#include <torch/torch.h>
#include <vector>

namespace ai::buffer {

class Buffer {
public:
  Buffer(int capacity, std::vector<size_t> observation_shape, int action_size);

private:
  int capacity_;
  torch::Tensor states_;
  torch::Tensor actions_;
  torch::Tensor rewards_;
  int indices_;
};

} // namespace ai::buffer