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
  observations_ = torch::zeros(buffer_observation_shape, torch::kByte);
  actions_ = torch::zeros({capacity_, action_size});
  rewards_ = torch::zeros({capacity_});
  terminals_ = torch::zeros({capacity_}, torch::kBool);
  truncations_ = torch::zeros({capacity_}, torch::kBool);
  indices_ = 0;
}

void Buffer::add(std::vector<unsigned char> observation, int action,
                 float reward, bool terminal, bool truncation) {
  observations_[indices_] =
      torch::from_blob(observation.data(), observation_shape_, torch::kByte);
  actions_[indices_] = torch::tensor(action);
  rewards_[indices_] = torch::tensor(reward);
  terminals_[indices_] = terminal;
  truncations_[indices_] = truncation;
  indices_ = (indices_ + 1) % capacity_;
}
} // namespace ai::buffer