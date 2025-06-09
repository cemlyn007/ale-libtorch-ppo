#include "ai/rollout.h"
#include <ale/ale_interface.hpp>
#include <ale/version.hpp>
#include <iostream>
#include <torch/torch.h>

int main(int argc, char **argv) {
  auto path = argv[1];
  torch::Tensor tensor = torch::eye(3);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    tensor = tensor.to(torch::kCUDA);
  } else {
    std::cout << "CUDA is not available! Training on CPU." << std::endl;
  }
  std::cout << tensor << std::endl;

  ai::rollout::Rollout rollout(std::filesystem::path(path), 128, 10, 1000, 4);
  for (size_t i = 0; i < 1000; i++) {
    std::cout << "Rollout " << i + 1 << " of 1000" << std::endl;
    rollout.rollout();
  }
  std::cout << "Success" << std::endl;
  return 0;
}
