#include "buffer.h"
#include "ai/gae.h"
#include <torch/torch.h>
namespace ai::buffer {
Buffer::Buffer(size_t total_environments, size_t capacity,
               std::vector<size_t> observation_shape, size_t action_size,
               const torch::Device &device)
    : device_(device), total_environments_(total_environments),
      capacity_(capacity), indices_(0) {

  // Prepend the batch size and total environments to the observation shape.
  std::vector<int64_t> observation_shape_(observation_shape.begin(),
                                          observation_shape.end());
  auto buffer_observation_shape = observation_shape_;
  buffer_observation_shape.insert(buffer_observation_shape.begin(), capacity_);
  buffer_observation_shape.insert(buffer_observation_shape.begin(),
                                  total_environments_);

  auto options = torch::TensorOptions(torch::kFloat32).device(device_);
  auto observation_options = options.dtype(torch::kByte);
  auto action_options = options.dtype(torch::kLong);
  auto state_options = options.dtype(torch::kBool);

  int64_t environments = static_cast<int64_t>(total_environments_);
  int64_t horizon = static_cast<int64_t>(capacity_);
  int64_t actions = static_cast<int64_t>(action_size);

  const std::array<int64_t, 2> scalar_shape = {environments, horizon};
  const std::array<int64_t, 3> logits_shape = {environments, horizon, actions};

  observations_ = torch::zeros(buffer_observation_shape, observation_options);
  actions_ = torch::zeros(scalar_shape, action_options);
  rewards_ = torch::zeros_like(actions_, options);
  terminals_ = torch::zeros_like(actions_, state_options);
  truncations_ = torch::zeros_like(terminals_);
  episode_starts_ = torch::zeros_like(terminals_);
  logits_ = torch::zeros(logits_shape, options);
  values_ = torch::zeros_like(rewards_);
  advantages_ = torch::zeros_like(values_);
  returns_ = torch::zeros_like(values_);
}

void Buffer::add(const torch::Tensor &observations,
                 const torch::Tensor &actions, const torch::Tensor &rewards,
                 const torch::Tensor &terminals,
                 const torch::Tensor &truncations,
                 const torch::Tensor &episode_starts,
                 const torch::Tensor &logits, const torch::Tensor &values) {
  const std::array<torch::indexing::TensorIndex, 2> indices = {
      torch::indexing::Slice(), indices_};
  observations_.index_put_(indices, observations.to(device_));
  actions_.index_put_(indices, actions.to(device_));
  rewards_.index_put_(indices, rewards.to(device_));
  terminals_.index_put_(indices, terminals.to(device_));
  truncations_.index_put_(indices, truncations.to(device_));
  episode_starts_.index_put_(indices, episode_starts.to(device_));
  logits_.index_put_(indices, logits.to(device_));
  values_.index_put_(indices, values.to(device_));
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