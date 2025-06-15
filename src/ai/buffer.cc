#include "buffer.h"
#include <torch/torch.h>
namespace ai::buffer {
Buffer::Buffer(size_t total_environments, size_t capacity,
               std::vector<size_t> observation_shape, size_t action_size)
    : total_environments_(total_environments), capacity_(capacity),
      indices_(0) {
  observation_shape_ =
      std::vector<int64_t>(observation_shape.begin(), observation_shape.end());
  auto buffer_observation_shape = observation_shape_;
  buffer_observation_shape.insert(buffer_observation_shape.begin(), capacity_);
  buffer_observation_shape.insert(buffer_observation_shape.begin(),
                                  total_environments_);
  observations_ = torch::zeros(buffer_observation_shape, torch::kByte);
  long total_environments_long = static_cast<long>(total_environments_);
  long capacity_long = static_cast<long>(capacity_);
  actions_ =
      torch::zeros({total_environments_long, capacity_long}, torch::kLong);
  rewards_ = torch::zeros({total_environments_long, capacity_long});
  terminals_ =
      torch::zeros({total_environments_long, capacity_long}, torch::kBool);
  truncations_ =
      torch::zeros({total_environments_long, capacity_long}, torch::kBool);
  episode_starts_ =
      torch::zeros({total_environments_long, capacity_long}, torch::kBool);
  logits_ = torch::zeros({total_environments_long, capacity_long,
                          static_cast<int64_t>(action_size)});
  values_ = torch::zeros({total_environments_long, capacity_long});
}

void Buffer::add(const torch::Tensor &observations,
                 const torch::Tensor &actions, const torch::Tensor &rewards,
                 const torch::Tensor &terminals,
                 const torch::Tensor &truncations,
                 const torch::Tensor &episode_starts,
                 const torch::Tensor &logits, const torch::Tensor &values) {
  observations_.index_put_({torch::indexing::Slice(), indices_}, observations);
  actions_.index_put_({torch::indexing::Slice(), indices_}, actions);
  rewards_.index_put_({torch::indexing::Slice(), indices_}, rewards);
  terminals_.index_put_({torch::indexing::Slice(), indices_}, terminals);
  truncations_.index_put_({torch::indexing::Slice(), indices_}, truncations);
  episode_starts_.index_put_({torch::indexing::Slice(), indices_},
                             episode_starts);
  logits_.index_put_({torch::indexing::Slice(), indices_}, logits);
  values_.index_put_({torch::indexing::Slice(), indices_}, values);
  indices_ = (indices_ + 1) % capacity_;
}
} // namespace ai::buffer