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
  auto expected_device = advantages.device();
  if (rewards.device() != expected_device) {
    throw std::invalid_argument("Rewards tensor must be on the same device as "
                                "advantages tensor.");
  }
  if (values.device() != expected_device) {
    throw std::invalid_argument("Values tensor must be on the same device as "
                                "advantages tensor.");
  }
  if (next_values.device() != expected_device) {
    throw std::invalid_argument("Next values tensor must be on the same device "
                                "as advantages tensor.");
  }
  if (terminals.device() != expected_device) {
    throw std::invalid_argument(
        "Terminals tensor must be on the same device as "
        "advantages tensor.");
  }
  if (truncations.device() != expected_device) {
    throw std::invalid_argument("Truncations tensor must be on the same device "
                                "as advantages tensor.");
  }
  if (episode_starts.device() != expected_device) {
    throw std::invalid_argument("Episode starts tensor must be on the same "
                                "device as advantages tensor.");
  }

  auto state_events = episode_starts.to(torch::kInt) +
                      terminals.to(torch::kInt) + truncations.to(torch::kInt);
  if ((state_events > 1).any().item<bool>())
    throw std::invalid_argument("Episode starts, terminals, and truncations "
                                "must be mutually exclusive.");

  auto total_environments = rewards.size(0);
  auto num_steps = rewards.size(1);
  auto last_advantages = torch::zeros(total_environments, rewards.options());
  torch::Tensor nv = next_values;
  for (int64_t i = num_steps - 1; i >= 0; --i) {
    auto advantage_reset = 0.0;
    auto advantage_running = rewards.select(1, i) + gamma * nv -
                             values.select(1, i) +
                             gamma * lambda * last_advantages;
    auto advantage_terminal = rewards.select(1, i) - values.select(1, i);
    auto advantage_truncation =
        rewards.select(1, i) + gamma * nv - values.select(1, i);

    auto advantage = torch::where(episode_starts.select(1, i), advantage_reset,
                                  advantage_running);
    advantage =
        torch::where(terminals.select(1, i), advantage_terminal, advantage);
    advantage =
        torch::where(truncations.select(1, i), advantage_truncation, advantage);

    advantages.select(1, i).copy_(advantage);

    last_advantages = advantage;
    nv = values.select(1, i);
  }
}

} // namespace ai::gae