#include "buffer.h"

#include <torch/torch.h>

#include "ai/gae.h"
namespace ai::buffer {
Buffer::Buffer(size_t total_environments, size_t capacity,
               std::vector<size_t> observation_shape, size_t action_size,
               const torch::Device &device)
    : device_(device),
      total_environments_(total_environments),
      capacity_(capacity) {
  observation_shape_ =
      std::vector<int64_t>(observation_shape.begin(), observation_shape.end());
  std::vector<int64_t> buffer_observation_shape = {
      static_cast<int64_t>(total_environments_),
      static_cast<int64_t>(capacity_)};
  buffer_observation_shape.insert(buffer_observation_shape.end(),
                                  observation_shape_.begin(),
                                  observation_shape_.end());
  int64_t total_environments_long = static_cast<int64_t>(total_environments_);
  int64_t capacity_long = static_cast<int64_t>(capacity_);

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

void Buffer::add_rows(
    int64_t env_start, int64_t env_count, int64_t time_index,
    const torch::Tensor &observations, const torch::Tensor &actions,
    const torch::Tensor &rewards, const torch::Tensor &terminals,
    const torch::Tensor &truncations, const torch::Tensor &episode_starts,
    const torch::Tensor &logits, const torch::Tensor &values) {
  auto rows = [&](torch::Tensor &t) {
    return t.narrow(0, env_start, env_count).select(1, time_index);
  };
  rows(observations_).copy_(observations);
  rows(actions_).copy_(actions);
  rows(rewards_).copy_(rewards);
  rows(terminals_).copy_(terminals);
  rows(truncations_).copy_(truncations);
  rows(episode_starts_).copy_(episode_starts);
  rows(logits_).copy_(logits);
  rows(values_).copy_(values);
  filled_rows_ += env_count;
}

Batch Buffer::get(const torch::Tensor &next_values, float discount,
                  float lambda) {
  if (filled_rows_ != static_cast<int64_t>(total_environments_ * capacity_))
    throw std::runtime_error("Buffer is not full, cannot compute GAE.");
  filled_rows_ = 0;

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
}  // namespace ai::buffer