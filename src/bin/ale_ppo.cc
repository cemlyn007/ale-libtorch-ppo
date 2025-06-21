#include "ai/ppo.h"
#include "ai/rollout.h"
#include "ai/video_recorder.h"
#include "ai/vision.h"
#include "tensorboard_logger.h"
#include <ale/ale_interface.hpp>
#include <ale/version.hpp>
#include <iostream>
#include <numeric>
#include <torch/nn.h>
#include <torch/torch.h>

struct Config {
  size_t total_environments = 512;
  size_t hidden_size = 128;
  const size_t action_size = 4;
  size_t horizon = 128;
  size_t max_steps = 108000;
  size_t frame_stack = 4;
  double learning_rate = 2.5e-4;
  double clip_param = 0.2;
  double value_loss_coef = 0.5;
  double entropy_coef = 0.001;
  long num_epochs = 4;
  long mini_batch_size = 2048;
  long num_mini_batches = 32; // num_mini_batches = horizon / mini_batch_size
  float gae_discount = 0.99f; // Discount factor for rewards
  float gae_lambda = 0.95f;   // GAE lambda for advantage estimation
  float max_gradient_norm = 0.5f; // Maximum norm for gradient clipping
  size_t num_rollouts = 1000000;
  size_t log_episode_frequency = 10;
};

static const Config config = Config();

struct Batch {
  torch::Tensor observations;
  torch::Tensor actions;
  torch::Tensor logits;
  torch::Tensor advantages;
  torch::Tensor returns;
  torch::Tensor masks;
};

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

struct Metrics {
  torch::Tensor loss;
  torch::Tensor clipped_losses;
  torch::Tensor value_losses;
  torch::Tensor entropies;
  torch::Tensor ratio;
  torch::Tensor total_losses;
  torch::Tensor advantages;
  torch::Tensor returns;
  torch::Tensor masks;
  torch::Tensor clipped_gradients;

  Metrics(const Config &config, const torch::Device &device) {
    auto options = torch::TensorOptions().device(device);
    loss = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        options);
    clipped_losses = torch::empty_like(loss);
    value_losses = torch::empty_like(loss);
    entropies = torch::empty_like(loss);
    ratio = torch::empty_like(loss);
    total_losses = torch::empty_like(loss);
    advantages = torch::empty_like(loss);
    returns = torch::empty_like(loss);
    masks = torch::empty(
        {config.num_epochs, config.num_mini_batches, config.mini_batch_size},
        options.dtype(torch::kBool));
    clipped_gradients =
        torch::empty({config.num_epochs, config.num_mini_batches});
  }
};

void log_data(TensorBoardLogger &logger, const ai::rollout::Log &log,
              const Metrics &metrics) {
  if (!log.episode_returns.empty()) {
    float mean_return = std::accumulate(log.episode_returns.begin(),
                                        log.episode_returns.end(), 0.0f) /
                        log.episode_returns.size();
    float mean_length = std::accumulate(log.episode_lengths.begin(),
                                        log.episode_lengths.end(), 0.0f) /
                        log.episode_lengths.size();
    logger.add_scalar("mean_episode_return", log.steps, mean_return);
    logger.add_scalar("mean_episode_length", log.steps, mean_length);
    logger.add_histogram("episode_returns", log.steps, log.episode_returns);
    logger.add_histogram("episode_lengths", log.steps, log.episode_lengths);
  }
  logger.add_scalar("mean_clipped_gradient", log.steps,
                    metrics.clipped_gradients.mean().item<float>());
  logger.add_scalar("mean_loss", log.steps, metrics.loss.mean().item<float>());
  logger.add_scalar("mean_clipped_loss", log.steps,
                    mean(metrics.clipped_losses, metrics.masks));
  logger.add_scalar("mean_value_loss", log.steps,
                    mean(metrics.value_losses, metrics.masks));
  logger.add_scalar("mean_entropy", log.steps,
                    mean(metrics.entropies, metrics.masks));
  logger.add_scalar("mean_ratio", log.steps,
                    mean(metrics.ratio, metrics.masks));
  logger.add_histogram("clipped_gradients", log.steps,
                       to_vector(metrics.clipped_gradients));
  logger.add_histogram("losses", log.steps,
                       gather(metrics.total_losses, metrics.masks));
  logger.add_histogram("clipped_losses", log.steps,
                       gather(metrics.clipped_losses, metrics.masks));
  logger.add_histogram("value_losses", log.steps,
                       gather(metrics.value_losses, metrics.masks));
  logger.add_histogram("entropies", log.steps,
                       gather(metrics.entropies, metrics.masks));
  logger.add_histogram("ratios", log.steps,
                       gather(metrics.ratio, metrics.masks));
  logger.add_histogram("advantages", log.steps,
                       gather(metrics.advantages, metrics.masks));
  logger.add_histogram("returns", log.steps,
                       gather(metrics.returns, metrics.masks));
}

void record(const std::filesystem::path &video_path, int64_t &episode,
            bool &recording, ai::video_recorder::VideoRecorder &recorder,
            ai::buffer::Batch &batch) {
  auto observations = batch.observations.index({0, torch::indexing::Slice(), 0})
                          .to(torch::kCPU);
  auto masks = batch.masks[0].to(torch::kCPU);
  for (size_t frame = 0; frame < config.horizon; ++frame) {
    bool new_episode = !masks[frame].item<bool>();
    if (new_episode) {
      ++episode;
    }
    if (episode % config.log_episode_frequency == 0) {
      if (new_episode && recording) {
        auto path = video_path / (std::to_string(episode) + ".mp4");
        recorder.complete(path);
      }
      const auto &observation = observations[frame];
      recorder.add(observation.data_ptr<uint8_t>());
      recording = true;
    }
  }
}

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
    {
      torch::NoGradGuard no_grad;
      if (x.device().is_cuda()) {
        x = x.to(torch::kFloat32);
      }
      x = ai::vision::resize_frame_stacked_grayscale_images(x);
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

void mini_batch_update(torch::Device &device, Network &network,
                       torch::optim::Adam &optimizer, Metrics &metrics,
                       const torch::Tensor &indices, const Batch &batch, long j,
                       long k) {
  {
    auto start = k * config.mini_batch_size;
    auto end = start + config.mini_batch_size;
    const auto &mini_indices = indices.slice(0, start, end);

    torch::Tensor mini_observations =
        batch.observations.index_select(0, mini_indices);
    torch::Tensor mini_actions = batch.actions.index_select(0, mini_indices);
    torch::Tensor mini_advantages =
        batch.advantages.index_select(0, mini_indices);
    torch::Tensor mini_logits = batch.logits.index_select(0, mini_indices);
    torch::Tensor mini_returns = batch.returns.index_select(0, mini_indices);
    torch::Tensor mini_masks = batch.masks.index_select(0, mini_indices);

    auto ppo_metrics =
        compute_loss(network, mini_observations, mini_actions, mini_advantages,
                     mini_logits, mini_returns, mini_masks, config.clip_param,
                     config.value_loss_coef, config.entropy_coef);
    optimizer.zero_grad();
    ppo_metrics.loss.backward();
    auto clipped_gradient = torch::nn::utils::clip_grad_norm_(
        network->parameters(), config.max_gradient_norm, 2.0, true);
    optimizer.step();

    const torch::indexing::TensorIndex indices({{j, k}});
    metrics.loss.index_put_(indices, ppo_metrics.loss.reshape({1}));
    metrics.clipped_losses.index_put_(indices, ppo_metrics.clipped_losses);
    metrics.value_losses.index_put_(indices, ppo_metrics.value_losses);
    metrics.entropies.index_put_(indices, ppo_metrics.entropies);
    metrics.ratio.index_put_(indices, ppo_metrics.ratio);
    metrics.total_losses.index_put_(indices, ppo_metrics.total_losses);
    metrics.advantages.index_put_(indices, mini_advantages);
    metrics.returns.index_put_(indices, mini_returns);
    metrics.masks.index_put_(indices, mini_masks);
    metrics.clipped_gradients.index_put_(indices, clipped_gradient);
  }
}

void train(torch::Device &device, Network &network,
           torch::optim::Adam &optimizer, Metrics &metrics,
           torch::Tensor &indices, ai::buffer::Batch &batch) {
  {
    network->train();
    auto observations = batch.observations.view({-1, batch.observations.size(2),
                                                 batch.observations.size(3),
                                                 batch.observations.size(4)});
    auto actions = batch.actions.ravel();
    auto advantages = batch.advantages.ravel();
    auto logits = batch.logits.view({-1, batch.logits.size(2)});
    auto returns = batch.returns.ravel();
    auto masks = batch.masks.ravel();
    auto batch =
        Batch(observations, actions, logits, advantages, returns, masks);
    for (long j = 0; j < config.num_epochs; j++) {
      torch::randperm_out(indices, observations.size(0));
      for (long k = 0; k < config.num_mini_batches; k++)
        mini_batch_update(device, network, optimizer, metrics, indices, batch,
                          j, k);
    }
  }
}

int main(int argc, char **argv) {
  const auto rom_path = argv[1];
  const auto logger_path = std::filesystem::path(argv[2]).replace_extension(
      "tfevents." +
      std::to_string(
          std::chrono::system_clock::now().time_since_epoch().count()));
  const auto video_path = std::filesystem::path(argv[3]);
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
  if (!std::filesystem::exists(video_path)) {
    std::filesystem::create_directories(video_path);
  }

  // For writing the images
  int64_t episode = -1;
  bool recording = false;
  auto recorder =
      ai::video_recorder::VideoRecorder(video_path, 1, 160, 210, 30);

  TensorBoardLogger logger(logger_path);
  Network network(config.hidden_size, config.action_size);
  initialize_weights(*network);
  network->to(device);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(config.learning_rate));
  ai::rollout::Rollout rollout(
      std::filesystem::path(rom_path), config.total_environments,
      config.horizon, config.max_steps, config.frame_stack,
      [&network, &device, action_size = config.action_size](
          const torch::Tensor &obs) -> ai::rollout::ActionResult {
        auto observations = device.is_cuda() ? obs.to(torch::kFloat32) : obs;
        auto output = network->forward(observations.to(device));
        auto logits = output.logits;
        auto probabilities = torch::nn::functional::softmax(logits, -1);
        auto actions = torch::multinomial(probabilities, 1, true);
        return {actions.ravel(),
                logits.reshape({-1, static_cast<long>(action_size)}),
                output.value.ravel()};
      },
      config.gae_discount, config.gae_lambda, device, 0);

  torch::Tensor indices =
      torch::empty(config.mini_batch_size * config.num_mini_batches,
                   torch::TensorOptions().dtype(torch::kLong).device(device));
  Metrics metrics(config, device);
  for (size_t i = 0; i < config.num_rollouts; i++) {
    std::cout << "Rollout " << i + 1 << " of " << config.num_rollouts
              << std::endl;
    ai::rollout::RolloutResult result;
    {
      network->eval();
      torch::NoGradGuard no_grad;
      result = rollout.rollout();
    }
    train(device, network, optimizer, metrics, indices, result.batch);
    log_data(logger, result.log, metrics);
    record(video_path, episode, recording, recorder, result.batch);
  }
  std::cout << "Success" << std::endl;
  return 0;
}
