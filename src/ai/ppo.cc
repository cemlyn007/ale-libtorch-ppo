#include "ppo.h"

namespace ai::ppo {
Metrics ppo_loss(const torch::Tensor &logits, const torch::Tensor &old_logits,
                 const torch::Tensor &actions, const torch::Tensor &advantages,
                 const torch::Tensor &values, const torch::Tensor &returns,
                 const torch::Tensor &masks, float clip_param,
                 float value_loss_coef, float entropy_coef) {
  auto log_probabilities = logits - torch::logsumexp(logits, -1, true);
  auto log_old_probabilities =
      old_logits - torch::logsumexp(old_logits, -1, true);
  auto action_indices = actions.unsqueeze(-1);

  auto ratio =
      torch::exp(log_probabilities.gather(-1, action_indices).squeeze(-1) -
                 log_old_probabilities.gather(-1, action_indices).squeeze(-1));
  auto unclipped_losses = ratio * advantages;
  auto clamped_losses =
      torch::clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages;
  auto clipped_losses = torch::max(-unclipped_losses, -clamped_losses);
  auto value_losses = 0.5 * torch::square(values - returns);

  auto entropies =
      -torch::sum(torch::exp(log_probabilities) * log_probabilities, -1);

  auto total_losses = clipped_losses + value_loss_coef * value_losses -
                      entropy_coef * entropies;
  return {torch::where(masks, total_losses, 0.0).sum() / masks.sum(),
          clipped_losses.detach(),
          value_losses.detach(),
          entropies.detach(),
          total_losses.detach(),
          ratio.detach(),
          masks.detach()};
}

} // namespace ai::ppo