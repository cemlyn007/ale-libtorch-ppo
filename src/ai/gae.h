#include <torch/torch.h>

namespace ai::gae {
void gae(torch::Tensor &advantages, const torch::Tensor &rewards,
         const torch::Tensor &values, const torch::Tensor &next_values,
         const torch::Tensor &terminals, const torch::Tensor &truncations,
         const torch::Tensor &episode_starts, float gamma, float lambda);
} // namespace ai::gae
