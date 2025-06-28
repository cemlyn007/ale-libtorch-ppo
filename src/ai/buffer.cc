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
  std::vector<int64_t> buffer_observation_shape = {
      static_cast<int64_t>(total_environments_),
      static_cast<int64_t>(capacity_)};
  buffer_observation_shape.insert(buffer_observation_shape.end(),
                                  observation_shape_.begin(),
                                  observation_shape_.end());
  long total_environments_long = static_cast<long>(total_environments_);
  long capacity_long = static_cast<long>(capacity_);

  auto options = torch::TensorOptions().device(device_);
  auto float_options = options.dtype(torch::kFloat32);
  auto bool_options = options.dtype(torch::kBool);
  auto long_options = options.dtype(torch::kLong);
  auto byte_options = options.dtype(torch::kByte);
  auto scalar_shape = {total_environments_long, capacity_long};
  auto logits_shape = {total_environments_long, capacity_long,
                       static_cast<int64_t>(action_size)};
  observations_ = torch::zeros(buffer_observation_shape, byte_options);
  actions_ = torch::zeros(scalar_shape, long_options);
  rewards_ = torch::zeros(scalar_shape, float_options);
  terminals_ = torch::zeros(scalar_shape, bool_options);
  truncations_ = torch::zeros(scalar_shape, bool_options);
  episode_starts_ = torch::zeros(scalar_shape, bool_options);
  logits_ = torch::zeros(logits_shape, float_options);
  values_ = torch::zeros(scalar_shape, float_options);
  advantages_ = torch::zeros(scalar_shape, float_options);
  returns_ = torch::zeros_like(advantages_);
}

void Buffer::add(const torch::Tensor &observations,
                 const torch::Tensor &actions, const torch::Tensor &rewards,
                 const torch::Tensor &terminals,
                 const torch::Tensor &truncations,
                 const torch::Tensor &episode_starts,
                 const torch::Tensor &logits, const torch::Tensor &values) {
  observations_.select(1, indices_).copy_(observations);
  actions_.select(1, indices_).copy_(actions);
  rewards_.select(1, indices_).copy_(rewards);
  terminals_.select(1, indices_).copy_(terminals);
  truncations_.select(1, indices_).copy_(truncations);
  episode_starts_.select(1, indices_).copy_(episode_starts);
  logits_.select(1, indices_).copy_(logits);
  values_.select(1, indices_).copy_(values);
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

  Batch batch{
      observations_, actions_, rewards_,    torch::logical_not(episode_starts_),
      logits_,       values_,  advantages_, returns_};
  return batch;
}
} // namespace ai::buffer