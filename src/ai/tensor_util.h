#pragma once
#include <torch/torch.h>

#include <numeric>
#include <stdexcept>
#include <vector>

namespace ai::tensor_util {

template <typename T>
float mean(const std::vector<T> &values) {
  if (values.empty()) throw std::invalid_argument("Values vector is empty.");
  return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

// Host copy of a tensor as a flat float vector (one device sync).
inline std::vector<float> to_vector(const torch::Tensor &tensor) {
  auto t = tensor.contiguous().to(torch::kCPU, torch::kFloat);
  float *data_ptr = t.data_ptr<float>();
  return std::vector<float>(data_ptr, data_ptr + t.numel());
}

inline std::vector<float> gather(const torch::Tensor &tensor,
                                 const torch::Tensor &mask) {
  return to_vector(tensor.masked_select(mask));
}

}  // namespace ai::tensor_util
