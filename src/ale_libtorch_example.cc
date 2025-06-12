#include "ai/ppo.h"
#include "ai/rollout.h"
#include "tensorboard_logger.h"
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
    if (x.device().is_cuda()) {
      x = x.to(torch::kFloat32);
    }
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

torch::Tensor
compute_loss(Network &network, const torch::Tensor &observations,
             const torch::Tensor &actions, const torch::Tensor &advantages,
             const torch::Tensor &old_logits, const torch::Tensor &returns,
             const torch::Tensor &masks, float clip_param = 0.2,
             float value_loss_coef = 0.5, float entropy_coef = 0.01) {
  // Compute the loss using the PPO loss function
  auto output = network->forward(observations);
  auto logits = output.logits;
  auto values = output.value;
  return ai::ppo::ppo_loss(logits, old_logits, actions, advantages, values,
                           returns, masks, clip_param, value_loss_coef,
                           entropy_coef);
}

int main(int argc, char **argv) {
  auto path = argv[1];
  auto logger_path = argv[2];
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  } else {
    std::cout << "CUDA is not available! Training on CPU." << std::endl;
  }

  TensorBoardLogger logger(logger_path);
  Network network(128, 4);
  network->to(device);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(0.001));
  ai::rollout::Rollout rollout(
      std::filesystem::path(path), 128, 10, 1000, 4,
      [&network,
       &device](const torch::Tensor &obs) -> ai::rollout::ActionResult {
        torch::NoGradGuard no_grad;
        auto observation = device.is_cuda() ? obs.to(torch::kFloat32) : obs;
        auto output = network->forward(observation.to(device).unsqueeze(0));
        auto logits = output.logits;
        auto probabilities = torch::nn::functional::softmax(
            logits, torch::nn::functional::SoftmaxFuncOptions(-1));
        auto action = torch::multinomial(probabilities, 1).item<int64_t>();
        return {static_cast<ale::Action>(action), logits.squeeze(),
                output.value.squeeze()};
      });
  for (size_t i = 0; i < 1000; i++) {
    std::cout << "Rollout " << i + 1 << " of 1000" << std::endl;
    auto batch = rollout.rollout();
    auto observations = batch.observations.to(device);
    auto actions = batch.actions.to(device);
    auto advantages = batch.advantages.to(device);
    auto logits = batch.logits.to(device);
    auto returns = batch.returns.to(device);
    auto masks = batch.masks.to(device);
    auto loss = compute_loss(network, observations, actions, advantages, logits,
                             returns, masks, 0.2, 0.5, 0.01);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    auto loss_value = loss.item<float>();
    std::cout << "Loss: " << loss_value << std::endl;
    logger.add_scalar("loss", i, loss_value);
  }
  std::cout << "Success" << std::endl;
  return 0;
}
