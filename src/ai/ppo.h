#include "torch/torch.h"

namespace ai::ppo {

struct Losses {
  torch::Tensor loss;
  torch::Tensor clipped_losses;
  torch::Tensor value_losses;
  torch::Tensor entropy_losses;
  torch::Tensor total_losses;
  torch::Tensor masks;
};

Losses ppo_loss(const torch::Tensor &logits, const torch::Tensor &old_logits,
                const torch::Tensor &actions, const torch::Tensor &advantages,
                const torch::Tensor &values, const torch::Tensor &returns,
                const torch::Tensor &masks, float clip_param,
                float value_loss_coef, float entropy_coef);

} // namespace ai::ppo