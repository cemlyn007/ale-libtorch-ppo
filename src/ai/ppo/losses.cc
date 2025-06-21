#include "ai/ppo/losses.h"

namespace ai::ppo::losses {
Metrics compute(const torch::Tensor &log_probabilities,
                const torch::Tensor &old_log_probabilities,
                const torch::Tensor &actions, const torch::Tensor &advantages,
                const torch::Tensor &values, const torch::Tensor &returns,
                const torch::Tensor &masks, float clip_param,
                float value_loss_coef, float entropy_coef) {
  auto action_indices = actions.unsqueeze(-1);
  auto clipped = clipped_surogate_objectives(
      log_probabilities.gather(-1, action_indices).squeeze(-1),
      old_log_probabilities.gather(-1, action_indices).squeeze(-1), advantages,
      clip_param);
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
                            const torch::Tensor &advantages, float clip_param) {
  auto ratios = torch::exp(log_probabilities - old_log_probabilities);
  auto clipped_ratios =
      torch::clamp(ratios, 1.0 - clip_param, 1.0 + clip_param);
  auto unclipped = ratios * advantages;
  auto clipped = clipped_ratios * advantages;
  return {torch::min(unclipped, clipped), ratios};
}

torch::Tensor compute_entropies(const torch::Tensor &log_probabilities) {
  return -torch::sum(torch::exp(log_probabilities) * log_probabilities, -1);
}

torch::Tensor normalize_logits(const torch::Tensor &logits) {
  return logits - torch::logsumexp(logits, -1, true);
}

} // namespace ai::ppo::losses