#include "ai/ppo/losses.h"

namespace ai::ppo::losses {
Metrics ppo_loss(const torch::Tensor &logits, const torch::Tensor &old_logits,
                 const torch::Tensor &actions, const torch::Tensor &advantages,
                 const torch::Tensor &values, const torch::Tensor &returns,
                 const torch::Tensor &masks, float clip_param,
                 float value_loss_coef, float entropy_coef) {
  auto log_probabilities = normalize_logits(logits);
  auto old_log_probabilities = normalize_logits(old_logits);
  auto clipped =
      clipped_surogate_objectives(log_probabilities, old_log_probabilities,
                                  actions, advantages, clip_param);
  auto value_losses = 0.5 * torch::square(values - returns);
  auto entropies = compute_entropies(log_probabilities);
  auto total_losses = -clipped.values + value_loss_coef * value_losses -
                      entropy_coef * entropies;
  return {torch::where(masks, total_losses, 0.0).sum() / masks.sum(),
          clipped.values.detach(),
          value_losses.detach(),
          entropies.detach(),
          total_losses.detach(),
          clipped.ratios.detach(),
          masks.detach()};
}

ClippedSurogateObjectivesResult
clipped_surogate_objectives(const torch::Tensor &log_probabilities,
                            const torch::Tensor &old_log_probabilities,
                            const torch::Tensor &actions,
                            const torch::Tensor &advantages, float clip_param) {
  auto action_indices = actions.unsqueeze(-1);
  auto ratio =
      torch::exp(log_probabilities.gather(-1, action_indices).squeeze(-1) -
                 old_log_probabilities.gather(-1, action_indices).squeeze(-1));
  auto unclipped = ratio * advantages;
  auto clipped =
      torch::clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages;
  return {torch::min(unclipped, clipped), ratio};
}

torch::Tensor compute_entropies(const torch::Tensor &log_probabilities) {
  return -torch::sum(torch::exp(log_probabilities) * log_probabilities, -1);
}

torch::Tensor normalize_logits(const torch::Tensor &logits) {
  return logits - torch::logsumexp(logits, -1, true);
}

} // namespace ai::ppo::losses