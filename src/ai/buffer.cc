#include "buffer.h"
#include "ai/gae.h"
#include <torch/torch.h>
namespace ai::buffer {
Buffer::Buffer(size_t total_environments, size_t capacity,
               std::vector<size_t> observation_shape, size_t action_size,
               const torch::Device &device)
    : device_(device), total_environments_(total_environments),
      capacity_(capacity), indices_(0) {
  observation_shape_ =
      std::vector<int64_t>(observation_shape.begin(), observation_shape.end());
  auto buffer_observation_shape = observation_shape_;
  buffer_observation_shape.insert(buffer_observation_shape.begin(), capacity_);
  buffer_observation_shape.insert(buffer_observation_shape.begin(),
                                  total_environments_);
  observations_ =
      torch::zeros(buffer_observation_shape,
                   torch::TensorOptions(torch::kByte).device(device_));
  long total_environments_long = static_cast<long>(total_environments_);
  long capacity_long = static_cast<long>(capacity_);

  auto float_options = torch::TensorOptions(torch::kFloat).device(device_);
  actions_ = torch::zeros({total_environments_long, capacity_long},
                          torch::TensorOptions(torch::kLong).device(device_));
  rewards_ =
      torch::zeros({total_environments_long, capacity_long}, float_options);
  terminals_ = torch::zeros({total_environments_long, capacity_long},
                            torch::TensorOptions(torch::kBool).device(device_));
  truncations_ =
      torch::zeros({total_environments_long, capacity_long},
                   torch::TensorOptions(torch::kBool).device(device_));
  episode_starts_ =
      torch::zeros({total_environments_long, capacity_long},
                   torch::TensorOptions(torch::kBool).device(device_));
  logits_ = torch::zeros({total_environments_long, capacity_long,
                          static_cast<int64_t>(action_size)},
                         float_options);
  values_ =
      torch::zeros({total_environments_long, capacity_long}, float_options);

  advantages_ =
      torch::zeros({total_environments_long, capacity_long},
                   torch::TensorOptions(torch::kFloat32).device(device_));
  returns_ = torch::zeros_like(advantages_);
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

Batch Buffer::get(const torch::Tensor &next_values, float discount,
                  float lambda) {
  // This function is a placeholder for the actual implementation.
  // It should return a reference to a Batch object containing the data
  // from the buffer, possibly using the final value function estimates
  // for the last observations.
  rewards_.clamp_(-1.0f, 1.0f);
  ai::gae::gae(advantages_, rewards_, values_, next_values, terminals_,
               truncations_, episode_starts_, discount, lambda);
  returns_.copy_(advantages_);
  returns_.add_(values_);
  episode_starts_.logical_not_();
  return {observations_, actions_, rewards_,    episode_starts_,
          logits_,       values_,  advantages_, returns_};
}
} // namespace ai::buffer