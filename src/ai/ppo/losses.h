#pragma once
#include <torch/torch.h>

namespace ai::ppo::losses {

struct Metrics {
  torch::Tensor loss;
  torch::Tensor clipped_losses;
  torch::Tensor value_losses;
  torch::Tensor entropies;
  torch::Tensor total_losses;
  torch::Tensor ratio;
  torch::Tensor masks;
};

struct ClippedSurrogateObjectivesResult {
  torch::Tensor values;
  torch::Tensor ratios;
};

Metrics compute(const torch::Tensor &log_probabilities,
                const torch::Tensor &old_log_probabilities,
                const torch::Tensor &actions, const torch::Tensor &advantages,
                const torch::Tensor &values, const torch::Tensor &returns,
                const torch::Tensor &masks, float clip_param,
                float value_loss_coef, float entropy_coef);

ClippedSurrogateObjectivesResult
clipped_surrogate_objectives(const torch::Tensor &log_probabilities,
                             const torch::Tensor &log_old_probabilities,
                             const torch::Tensor &advantages, float clip_param);

// Requires log probabilities.
torch::Tensor compute_entropies(const torch::Tensor &);

// Accepts unnormalized logits.
torch::Tensor normalize_logits(const torch::Tensor &);

} // namespace ai::ppo::losses