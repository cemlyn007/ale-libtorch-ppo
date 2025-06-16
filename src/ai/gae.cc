#include "gae.h"

namespace ai::gae {
void gae(torch::Tensor &advantages, const torch::Tensor &rewards,
         const torch::Tensor &values, const torch::Tensor &next_values,
         const torch::Tensor &terminals, const torch::Tensor &truncations,
         const torch::Tensor &episode_starts, float gamma, float lambda) {
  if (rewards.dim() != 2 || values.dim() != 2 || next_values.dim() != 1 ||
      terminals.dim() != 2 || truncations.dim() != 2 ||
      episode_starts.dim() != 2) {
    throw std::invalid_argument(
        "All input tensors must be 2D except next_values which must be 1D.");
  }
  if (rewards.size(0) != values.size(0) ||
      rewards.size(0) != terminals.size(0) ||
      rewards.size(0) != truncations.size(0) ||
      rewards.size(0) != episode_starts.size(0) ||
      values.size(0) != next_values.size(0)) {
    throw std::invalid_argument(
        "Input tensors must have compatible dimensions.");
  }
  auto expected_device = rewards.device();
  if (values.device() != expected_device) {
    throw std::invalid_argument("Values tensor must be on the same device as "
                                "rewards tensor.");
  }
  if (next_values.device() != expected_device) {
    throw std::invalid_argument("Next values tensor must be on the same device "
                                "as rewards tensor.");
  }
  if (terminals.device() != expected_device) {
    throw std::invalid_argument(
        "Terminals tensor must be on the same device as "
        "rewards tensor.");
  }
  if (truncations.device() != expected_device) {
    throw std::invalid_argument("Truncations tensor must be on the same device "
                                "as rewards tensor.");
  }
  if (episode_starts.device() != expected_device) {
    throw std::invalid_argument("Episode starts tensor must be on the same "
                                "device as rewards tensor.");
  }
  auto total_environments = rewards.size(0);
  auto num_steps = rewards.size(1);
  auto last_advantages = torch::zeros(total_environments, rewards.options());
  torch::Tensor nv = next_values;
  for (int64_t i = num_steps - 1; i >= 0; --i) {
    advantages.index_put_(
        {torch::indexing::Slice(), i},
        torch::where(episode_starts.index({torch::indexing::Slice(), i}), 0.0,
                     rewards.index({torch::indexing::Slice(), i}) + gamma * nv -
                         values.index({torch::indexing::Slice(), i}) +
                         gamma * lambda * last_advantages));
    advantages.index_put_(
        {torch::indexing::Slice(), i},
        torch::where(terminals.index({torch::indexing::Slice(), i}),
                     rewards.index({torch::indexing::Slice(), i}) -
                         values.index({torch::indexing::Slice(), i}),
                     advantages.index({torch::indexing::Slice(), i})));

    advantages.index_put_(
        {torch::indexing::Slice(), i},
        torch::where(truncations.index({torch::indexing::Slice(), i}),
                     rewards.index({torch::indexing::Slice(), i}) + gamma * nv -
                         values.index({torch::indexing::Slice(), i}),
                     advantages.index({torch::indexing::Slice(), i})));
    last_advantages = advantages.index({torch::indexing::Slice(), i});
    nv = values.index({torch::indexing::Slice(), i});
  }
}

} // namespace ai::gae