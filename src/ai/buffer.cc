#include "buffer.h"
#include <torch/torch.h>
namespace ai::buffer {
Buffer::Buffer(int capacity, std::vector<size_t> observation_shape,
               int action_size)
    : capacity_(capacity) {
  observation_shape.insert(observation_shape.begin(), capacity_);
  states_ = torch::zeros(
      std::vector<int64_t>(observation_shape.begin(), observation_shape.end()));
  actions_ = torch::zeros({capacity_, action_size});
  rewards_ = torch::zeros({capacity_});
  indices_ = 0;
}
} // namespace ai::buffer