#include "buffer.h"
#include <torch/torch.h>
namespace ai::buffer {
Buffer::Buffer(size_t capacity, std::vector<size_t> observation_shape,
               size_t action_size)
    : capacity_(capacity), indices_(0) {
  observation_shape_ =
      std::vector<int64_t>(observation_shape.begin(), observation_shape.end());
  auto buffer_observation_shape = observation_shape_;
  buffer_observation_shape.insert(buffer_observation_shape.begin(), capacity_);
  observations_ = torch::zeros(buffer_observation_shape, torch::kByte);
  long capacity_long = static_cast<long>(capacity_);
  actions_ = torch::zeros({capacity_long});
  rewards_ = torch::zeros({capacity_long});
  terminals_ = torch::zeros({capacity_long}, torch::kBool);
  truncations_ = torch::zeros({capacity_long}, torch::kBool);
  logits_ = torch::zeros({capacity_long, static_cast<int64_t>(action_size)});
  values_ = torch::zeros({capacity_long});
  indices_ = 0;
}

void Buffer::add(std::vector<unsigned char> observation, int action,
                 float reward, bool terminal, bool truncation,
                 torch::Tensor logits, torch::Tensor value) {
  observations_[indices_] =
      torch::from_blob(observation.data(), observation_shape_, torch::kByte);
  actions_[indices_] = torch::tensor(action);
  rewards_[indices_] = torch::tensor(reward);
  terminals_[indices_] = terminal;
  truncations_[indices_] = truncation;
  logits_[indices_] = logits;
  values_[indices_] = value;
  indices_ = (indices_ + 1) % capacity_;
}
} // namespace ai::buffer