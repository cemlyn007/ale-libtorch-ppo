#include "buffer.h"
#include <torch/torch.h>
namespace ai::buffer {
Buffer::Buffer(int capacity, std::vector<size_t> observation_shape,
               int action_size)
    : capacity_(capacity), indices_(0) {
  observation_shape_ =
      std::vector<int64_t>(observation_shape.begin(), observation_shape.end());
  auto buffer_observation_shape = observation_shape_;
  buffer_observation_shape.insert(buffer_observation_shape.begin(), capacity_);
  states_ = torch::zeros(buffer_observation_shape, torch::kByte);
  actions_ = torch::zeros({capacity_, action_size});
  rewards_ = torch::zeros({capacity_});
  indices_ = 0;
}
} // namespace ai::buffer