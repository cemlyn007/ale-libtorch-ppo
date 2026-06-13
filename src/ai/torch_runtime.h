#pragma once
#include <torch/torch.h>

#include <cstdint>

namespace ai::torch_runtime {

// CUDA when available, MPS on Apple, otherwise CPU. Logs the choice.
torch::Device select_device();

// Seed torch and force deterministic algorithms (throws on non-deterministic
// ops). Also sets CUBLAS_WORKSPACE_CONFIG so cuBLAS GEMMs are reproducible.
void enable_determinism(uint64_t seed);

}  // namespace ai::torch_runtime
