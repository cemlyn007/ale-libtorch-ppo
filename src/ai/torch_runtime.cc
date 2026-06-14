#include "ai/torch_runtime.h"

#include <spdlog/spdlog.h>

#include <cstdlib>

namespace ai::torch_runtime {

torch::Device select_device() {
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    spdlog::info("CUDA is available! Training on GPU.");
    device = torch::Device(torch::kCUDA);
  } else {
    spdlog::warn("CUDA is not available! Training on CPU.");
  }
#ifdef __APPLE__
  device = torch::Device(torch::kMPS);
#endif
  return device;
}

void enable_determinism(uint64_t seed) {
  // As per the logged warning by LibTorch: "Warning: Deterministic behavior was
  // enabled with either `torch.use_deterministic_algorithms(True)` or
  // `at::Context::setDeterministicAlgorithms(true)`, but this operation is not
  // deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable
  // deterministic behavior in this case, you must set an environment variable
  // before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
  // CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
  // https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
  // (function alertCuBLASConfigNotDeterministic)"
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);

  torch::manual_seed(seed);

  // Enable deterministic algorithms, throw errors for non-deterministic
  // operations
  torch::globalContext().setDeterministicAlgorithms(true, true);

  // If using CUDA, ensure CuDNN is deterministic
  if (torch::cuda::is_available()) {
    torch::globalContext().setDeterministicCuDNN(true);
  }

  // Optionally, enable filling uninitialized memory for additional determinism
  torch::globalContext().setDeterministicFillUninitializedMemory(true);
}

}  // namespace ai::torch_runtime
