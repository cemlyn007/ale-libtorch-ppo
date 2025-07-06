#include "ai/ppo/train.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace ai::ppo::train {

torch::Tensor clip_grad_norm_(const std::vector<torch::Tensor> &parameters,
                              double max_norm) {
  std::vector<torch::Tensor> params_with_grad;

  for (const auto &param : parameters) {
    auto &grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(param);
    }
  }

  if (params_with_grad.empty()) {
    return torch::full({}, 0.0, parameters[0].options());
  }

  torch::Tensor total_norm_tensor;

  std::vector<torch::Tensor> norms;
  norms.reserve(params_with_grad.size());

  const double norm_type = 2.0; // L2 norm
  for (const auto &param : params_with_grad) {
    norms.emplace_back(param.grad().data().norm(norm_type));
  }
  total_norm_tensor =
      (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(norm_type);

  auto clip_coef = max_norm / (total_norm_tensor + 1e-6);
  auto clip_coef_clamped =
      torch::clamp(clip_coef, c10::nullopt /* min */, 1.0 /* max */);
  for (auto &param : params_with_grad) {
    param.grad().data().mul_(clip_coef_clamped);
  }
  return total_norm_tensor;
}

void stream_sync(at::cuda::CUDAStream &dependency,
                 at::cuda::CUDAStream &dependent) {
  at::cuda::CUDAEvent cuda_ev;
  cuda_ev.record(dependency);
  cuda_ev.block(dependent);
}

void train_cuda_graph(at::cuda::CUDAGraph &graph) { graph.replay(); }

} // namespace ai::ppo::train