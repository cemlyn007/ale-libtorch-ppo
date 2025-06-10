#include "gae.h"

namespace ai::gae {
torch::Tensor gae(const torch::Tensor &rewards, const torch::Tensor &values,
                  const torch::Tensor &next_values,
                  const torch::Tensor &terminals,
                  const torch::Tensor &truncations,
                  const torch::Tensor &episode_starts, float gamma,
                  float lambda) {
  auto num_steps = rewards.size(0);
  auto advantages = torch::zeros_like(rewards);
  auto last_advantages = torch::zeros_like(rewards[0]);
  torch::Tensor nv = next_values;
  for (int64_t i = num_steps - 1; i >= 0; --i) {
    advantages[i] = torch::where(
        episode_starts[i], torch::zeros_like(rewards[i]),
        rewards[i] + gamma * nv - values[i] + gamma * lambda * last_advantages);
    advantages[i] =
        torch::where(terminals[i], rewards[i] - values[i], advantages[i]);
    advantages[i] = torch::where(
        truncations[i], rewards[i] + gamma * nv - values[i], advantages[i]);
    last_advantages = advantages[i];
    nv = values[i];
  }
  return advantages;
}

} // namespace ai::gae