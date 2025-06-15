#include "ai/ppo.h"
#include "ai/rollout.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "tensorboard_logger.h"
#include <ale/ale_interface.hpp>
#include <ale/version.hpp>
#include <iostream>
#include <numeric>
#include <torch/nn.h>
#include <torch/torch.h>

struct Config {
  size_t total_environments = 32;
  size_t hidden_size = 32;
  size_t action_size = 4;
  size_t horizon = 128;
  size_t max_steps = 108000;
  size_t frame_stack = 4;
  double learning_rate = 2.5e-4;
  double clip_param = 0.2;
  double value_loss_coef = 0.5;
  double entropy_coef = 0.001;
  long num_epochs = 4;
  long mini_batch_size = 256;
  long num_mini_batches = 16; // num_mini_batches = horizon / mini_batch_size
  float gae_gamma = 0.99f;    // Discount factor for rewards
  float gae_lambda = 0.95f;   // GAE lambda for advantage estimation
  float max_gradient_norm = 0.5f; // Maximum norm for gradient clipping
  size_t num_rollouts = 1000000;
  bool log_images = false;
};

static const Config config = Config();

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
    {
      torch::NoGradGuard no_grad;
      auto batch_size = x.size(0);
      auto frame_stack = x.size(1);
      x = x.reshape({x.size(0) * x.size(1), 1, x.size(2), x.size(3)});
      x = torch::nn::functional::interpolate(
          x, torch::nn::functional::InterpolateFuncOptions()
                 .size(std::vector<int64_t>({84, 84}))
                 .mode(torch::kBilinear)
                 .align_corners(false));
      x = x.reshape({batch_size, frame_stack, 84, 84});
      x = x.to(torch::kFloat32) / 255.0;
    }
    x = sequential->forward(x);
    auto logits = action_head->forward(x);
    auto value = value_head->forward(x).squeeze(-1);
    return {logits, value};
  }

  torch::nn::Sequential sequential;
  torch::nn::Linear action_head, value_head;
};
TORCH_MODULE(Network);

float mean(const torch::Tensor &tensor, const torch::Tensor &mask) {
  auto masked_tensor = tensor.masked_select(mask);
  return masked_tensor.mean().item<float>();
}

std::vector<float> gather(const torch::Tensor &tensor,
                          const torch::Tensor &mask) {
  auto t =
      tensor.masked_select(mask).contiguous().to(torch::kCPU, torch::kFloat);
  float *data_ptr = t.data_ptr<float>();
  return std::vector<float>(data_ptr, data_ptr + t.numel());
}

std::vector<float> to_vector(const torch::Tensor &tensor) {
  auto t = tensor.contiguous().to(torch::kCPU, torch::kFloat);
  float *data_ptr = t.data_ptr<float>();
  return std::vector<float>(data_ptr, data_ptr + t.numel());
}

void initialize_weights(torch::nn::Module &module) {
  for (auto &submodule : module.children()) {
    if (auto linear = dynamic_cast<torch::nn::LinearImpl *>(submodule.get())) {

      linear->weight.data().normal_(
          0.0, 1.0 / std::sqrt(linear->options.in_features()));
      if (linear->bias.defined()) {
        linear->bias.data().fill_(1.0 /
                                  std::sqrt(linear->options.in_features()));
      }
    } else {
      initialize_weights(*submodule);
    }
  }
}

ai::ppo::Metrics
compute_loss(Network &network, const torch::Tensor &observations,
             const torch::Tensor &actions, const torch::Tensor &advantages,
             const torch::Tensor &old_logits, const torch::Tensor &returns,
             const torch::Tensor &masks, float clip_param,
             float value_loss_coef, float entropy_coef) {
  auto output = network->forward(observations);
  auto logits = output.logits;
  auto values = output.value;
  return ai::ppo::ppo_loss(logits, old_logits, actions, advantages, values,
                           returns, masks, clip_param, value_loss_coef,
                           entropy_coef);
}

int main(int argc, char **argv) {
  auto path = argv[1];
  auto logger_path = std::filesystem::path(argv[2]);
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  } else {
    std::cout << "CUDA is not available! Training on CPU." << std::endl;
  }

  if (!std::filesystem::exists(logger_path.parent_path())) {
    std::filesystem::create_directories(logger_path.parent_path());
  }

  auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
  logger_path =
      logger_path.replace_extension("tfevents." + std::to_string(timestamp));

  TensorBoardLogger logger(logger_path);
  int64_t action_size = 4;
  Network network(config.hidden_size, action_size);
  initialize_weights(*network);
  network->to(device);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(config.learning_rate));
  ai::rollout::Rollout rollout(
      std::filesystem::path(path), config.total_environments, config.horizon,
      config.max_steps, config.frame_stack,
      [&network, &device,
       &action_size](const torch::Tensor &obs) -> ai::rollout::ActionResult {
        torch::NoGradGuard no_grad;
        auto observations = device.is_cuda() ? obs.to(torch::kFloat32) : obs;
        auto output = network->forward(observations.to(device));
        auto logits = output.logits;
        auto probabilities = torch::nn::functional::softmax(
            logits, torch::nn::functional::SoftmaxFuncOptions(-1));
        auto actions = torch::multinomial(probabilities, 1, true);
        return {actions.ravel(), logits.reshape({-1, action_size}),
                output.value.ravel()};
      },
      config.gae_gamma, config.gae_lambda);

  for (size_t i = 0; i < config.num_rollouts; i++) {
    std::cout << "Rollout " << i + 1 << " of " << config.num_rollouts
              << std::endl;
    auto result = rollout.rollout();
    auto batch = result.batch;
    auto log = result.log;

    // Display episode returns and lengths
    if (!log.episode_returns.empty()) {
      float mean_return = std::accumulate(log.episode_returns.begin(),
                                          log.episode_returns.end(), 0.0f) /
                          log.episode_returns.size();
      float mean_length = std::accumulate(log.episode_lengths.begin(),
                                          log.episode_lengths.end(), 0.0f) /
                          log.episode_lengths.size();
      // Log to tensorboard
      logger.add_scalar("mean_episode_return", log.steps, mean_return);
      logger.add_scalar("mean_episode_length", log.steps, mean_length);
      logger.add_histogram("episode_returns", log.steps, log.episode_returns);
      logger.add_histogram("episode_lengths", log.steps, log.episode_lengths);
    }

    auto batch_observations =
        batch.observations
            .reshape({-1, batch.observations.size(2),
                      batch.observations.size(3), batch.observations.size(4)})
            .to(device);
    auto batch_actions = batch.actions.ravel().to(device);
    auto batch_advantages = batch.advantages.ravel().to(device);
    auto batch_logits =
        batch.logits.reshape({-1, batch.logits.size(2)}).to(device);
    auto batch_returns = batch.returns.ravel().to(device);
    auto batch_masks = batch.masks.ravel().to(device);

    if (batch_observations.size(0) !=
        config.mini_batch_size * config.num_mini_batches) {
      throw std::runtime_error(
          "Batch size is not divisible by mini-batch size");
    }

    auto clipped_gradients =
        torch::empty({config.num_epochs, config.num_mini_batches});

    auto metric_loss = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));
    auto metric_clipped_losses = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));
    auto metric_value_losses = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));
    auto metric_entropies = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));
    auto metric_ratio = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));
    auto metric_total_losses = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));
    auto metric_advantages = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));
    auto metric_returns = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().device(device));

    auto metric_masks = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        torch::TensorOptions().dtype(torch::kBool).device(device));

    for (long j = 0; j < config.num_epochs; j++) {
      torch::Tensor indices = torch::randperm(
          batch_observations.size(0),
          torch::TensorOptions().dtype(torch::kLong).device(device));
      for (long k = 0; k < config.num_mini_batches; k++) {
        auto start = k * config.mini_batch_size;
        auto end = start + config.mini_batch_size;
        auto mini_batch_indices = indices.slice(0, start, end);

        auto observations =
            batch_observations.index_select(0, mini_batch_indices).to(device);
        auto actions =
            batch_actions.index_select(0, mini_batch_indices).to(device);
        auto advantages =
            batch_advantages.index_select(0, mini_batch_indices).to(device);
        auto logits =
            batch_logits.index_select(0, mini_batch_indices).to(device);
        auto returns =
            batch_returns.index_select(0, mini_batch_indices).to(device);
        auto masks = batch_masks.index_select(0, mini_batch_indices).to(device);

        auto metrics = compute_loss(
            network, observations, actions, advantages, logits, returns, masks,
            config.clip_param, config.value_loss_coef, config.entropy_coef);

        optimizer.zero_grad();
        metrics.loss.backward();
        auto clipped_gradient = torch::nn::utils::clip_grad_norm_(
            network->parameters(), config.max_gradient_norm, 2.0, true);
        optimizer.step();

        clipped_gradients.index_put_({{j, k}}, clipped_gradient);
        metric_loss.index_put_({{j, k}}, metrics.loss.reshape({1}));
        metric_clipped_losses.index_put_({{j, k}}, metrics.clipped_losses);
        metric_value_losses.index_put_({{j, k}}, metrics.value_losses);
        metric_entropies.index_put_({{j, k}}, metrics.entropies);
        metric_ratio.index_put_({{j, k}}, metrics.ratio);
        metric_total_losses.index_put_({{j, k}}, metrics.total_losses);
        metric_advantages.index_put_({{j, k}}, advantages);
        metric_returns.index_put_({{j, k}}, returns);
        metric_masks.index_put_({{j, k}}, masks);
      }
    }

    logger.add_scalar("mean_clipped_gradient", log.steps,
                      clipped_gradients.mean().item<float>());
    logger.add_scalar("mean_loss", log.steps, metric_loss.mean().item<float>());
    logger.add_scalar("mean_clipped_loss", log.steps,
                      mean(metric_clipped_losses, metric_masks));
    logger.add_scalar("mean_value_loss", log.steps,
                      mean(metric_value_losses, metric_masks));
    logger.add_scalar("mean_entropy", log.steps,
                      mean(metric_entropies, metric_masks));
    logger.add_scalar("mean_ratio", log.steps,
                      mean(metric_ratio, metric_masks));
    // Histogram
    logger.add_histogram("actions", log.steps,
                         gather(batch_actions, batch_masks));
    logger.add_histogram(
        "probabilities", log.steps,
        gather(torch::nn::functional::softmax(
                   batch_logits, torch::nn::functional::SoftmaxFuncOptions(-1)),
               batch_masks.unsqueeze(1)));
    logger.add_histogram("clipped_gradients", log.steps,
                         to_vector(clipped_gradients));
    logger.add_histogram("losses", log.steps,
                         gather(metric_total_losses, metric_masks));
    logger.add_histogram("clipped_losses", log.steps,
                         gather(metric_clipped_losses, metric_masks));
    logger.add_histogram("value_losses", log.steps,
                         gather(metric_value_losses, metric_masks));
    logger.add_histogram("entropies", log.steps,
                         gather(metric_entropies, metric_masks));
    logger.add_histogram("ratios", log.steps,
                         gather(metric_ratio, metric_masks));
    logger.add_histogram("advantages", log.steps,
                         gather(metric_advantages, metric_masks));
    logger.add_histogram("returns", log.steps,
                         gather(metric_returns, metric_masks));

    if (config.log_images) {
      for (int64_t index = 0; index < batch_observations.size(0); ++index) {
        auto observation = batch_observations.index({index, 0}).to(torch::kCPU);
        auto numel = observation.numel();
        std::vector<unsigned char> img_data(numel);
        std::memcpy(img_data.data(), observation.data_ptr<uint8_t>(),
                    numel * sizeof(uint8_t));
        int width = 160, height = 210, channels = 1;
        std::string png_data;
        auto write_func = [](void *context, void *data, int size) {
          auto *str = static_cast<std::string *>(context);
          str->append(static_cast<char *>(data), size);
        };
        stbi_write_png_to_func(write_func, &png_data, width, height, channels,
                               img_data.data(), width * channels);
        logger.add_image("observation", log.steps - config.horizon + index,
                         png_data, 210, 160, 1);
      }
    }
  }
  std::cout << "Success" << std::endl;
  return 0;
}
