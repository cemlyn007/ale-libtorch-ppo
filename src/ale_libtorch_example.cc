#include "ai/rollout.h"
#include <ale/ale_interface.hpp>
#include <ale/version.hpp>
#include <iostream>
#include <torch/nn.h>
#include <torch/torch.h>

struct ForwardResult {
  torch::Tensor logits;
  torch::Tensor value;
};

struct NetworkImpl : torch::nn::Module {
  NetworkImpl(size_t hidden_size, size_t action_size)
      : sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 32, 8).stride(4)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
            torch::nn::ReLU(), torch::nn::Flatten(),
            torch::nn::Linear(64 * 7 * 7, hidden_size), torch::nn::ReLU()),
        action_head(torch::nn::Linear(hidden_size, action_size)),
        value_head(torch::nn::Linear(hidden_size, 1)) {
    register_module("sequential", sequential);
    register_module("action_head", action_head);
    register_module("value_head", value_head);
  }

  ForwardResult forward(torch::Tensor x) {
    x = torch::nn::functional::interpolate(
        x, torch::nn::functional::InterpolateFuncOptions()
               .size(std::vector<int64_t>({84, 84}))
               .mode(torch::kBilinear)
               .align_corners(false));
    x = x.to(torch::kFloat32) / 255.0;
    x = sequential->forward(x);
    auto logits = action_head->forward(x);
    auto value = value_head->forward(x);
    return {logits, value};
  }

  torch::nn::Sequential sequential;
  torch::nn::Linear action_head, value_head;
};
TORCH_MODULE(Network);

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

  Network network(128, 4);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(0.001));

  auto action_selector = [&network](const torch::Tensor &obs) -> ale::Action {
    torch::NoGradGuard no_grad;
    auto output = network->forward(obs.unsqueeze(0));
    auto logits = output.logits;
    auto action_idx = logits.argmax(-1).item<int>();
    return static_cast<ale::Action>(action_idx);
  };

  ai::rollout::Rollout rollout(std::filesystem::path(path), 128, 10, 1000, 4,
                               action_selector);
  for (size_t i = 0; i < 1000; i++) {
    std::cout << "Rollout " << i + 1 << " of 1000" << std::endl;
    auto batch = rollout.rollout();
    optimizer.zero_grad();
  }
  std::cout << "Success" << std::endl;
  return 0;
}
