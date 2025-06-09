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
  ale::ALEInterface ale;
  std::cout << "Initializing Arcade Learning Environment..." << std::endl;
  ale.loadROM(path);
  std::cout << "ALE initialized!" << std::endl;
  std::cout << "Number of available actions: "
            << ale.getMinimalActionSet().size() << std::endl;
  ale.reset_game();
  std::cout << "Game reset!" << std::endl;
  return 0;
}
