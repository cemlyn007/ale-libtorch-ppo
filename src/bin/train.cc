#include "ai/ppo/train.h"
#include "ai/ppo/losses.h"
#include "ai/rollout.h"
#include "ai/vision.h"
#include "tensorboard_logger.h"
#include <ale/ale_interface.hpp>
#include <ale/version.hpp>
#include <iostream>
#include <numeric>
#include <torch/nn.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

const bool USE_CUDA_GRAPH = false;
const bool PREPROCESSED_GRAYSCALE = true;
const bool ANNEAL_ENTROPY_COEFFICIENT = false;

// This is being used for annealing the entropy coefficient
//  based on the average return.
static double sum = 0;
static double count = 1;

double get_average_return() { return sum / count; }

double get_annealed_entropy_coef(double entropy_coef) {
  // This is a simple annealing function that decreases the entropy
  // coefficient as the average return increases.
  // 864 is the maximum episodic return for playing breakout.
  if (!ANNEAL_ENTROPY_COEFFICIENT) {
    return entropy_coef;
  }
  return entropy_coef * (864.0 - get_average_return()) / 864.0;
}

struct Config {
  size_t total_environments;
  size_t hidden_size;
  const size_t action_size = 4;
  size_t horizon;
  size_t max_steps;
  size_t frame_stack;
  double learning_rate;
  float clip_param;
  float value_loss_coef;
  float entropy_coef;
  long num_epochs;
  long mini_batch_size;
  long num_mini_batches;
  float gae_discount;
  float gae_lambda;
  float max_gradient_norm;
  size_t num_rollouts;
  size_t log_episode_frequency;
  size_t num_workers;
  size_t worker_batch_size;
  size_t frame_skip;
  // Some games like breakout have a maximum return
  // which should be used to reset the environment.
  float max_return;
  // It is faster to record using the observation.
  // However the observation may be in grayscale.
  bool record_observation;
};

google::protobuf::Value get_value(double value) {
  google::protobuf::Value val;
  val.set_number_value(value);
  return val;
}

google::protobuf::Value get_bool_value(bool value) {
  google::protobuf::Value val;
  val.set_bool_value(value);
  return val;
}

std::map<std::string, google::protobuf::Value>
get_parameters(const Config &config) {
  std::map<std::string, google::protobuf::Value> hparams;
  hparams["total_environments"] = get_value(config.total_environments);
  hparams["hidden_size"] = get_value(config.hidden_size);
  hparams["action_size"] = get_value(config.action_size);
  hparams["horizon"] = get_value(config.horizon);
  hparams["max_steps"] = get_value(config.max_steps);
  hparams["frame_stack"] = get_value(config.frame_stack);
  hparams["learning_rate"] = get_value(config.learning_rate);
  hparams["clip_param"] = get_value(config.clip_param);
  hparams["value_loss_coef"] = get_value(config.value_loss_coef);
  hparams["entropy_coef"] = get_value(config.entropy_coef);
  hparams["num_epochs"] = get_value(config.num_epochs);
  hparams["mini_batch_size"] = get_value(config.mini_batch_size);
  hparams["num_mini_batches"] = get_value(config.num_mini_batches);
  hparams["gae_discount"] = get_value(config.gae_discount);
  hparams["gae_lambda"] = get_value(config.gae_lambda);
  hparams["max_gradient_norm"] = get_value(config.max_gradient_norm);
  hparams["num_rollouts"] = get_value(config.num_rollouts);
  hparams["log_episode_frequency"] = get_value(config.log_episode_frequency);
  hparams["num_workers"] = get_value(config.num_workers);
  hparams["worker_batch_size"] = get_value(config.worker_batch_size);
  hparams["frame_skip"] = get_value(config.frame_skip);
  hparams["max_return"] = get_value(config.max_return);
  hparams["record_observation"] = get_bool_value(config.record_observation);
  return hparams;
}

Config load_config(const std::filesystem::path &path) {
  Config config;
  YAML::Node node = YAML::LoadFile(path.string());
  config.total_environments = node["total_environments"].as<size_t>(512);
  config.hidden_size = node["hidden_size"].as<size_t>(512);
  config.horizon = node["horizon"].as<size_t>(128);
  config.max_steps = node["max_steps"].as<size_t>(108000);
  config.frame_stack = node["frame_stack"].as<size_t>(4);
  config.learning_rate = node["learning_rate"].as<double>(2.5e-4);
  config.clip_param = node["clip_param"].as<float>(0.1f);
  config.value_loss_coef = node["value_loss_coef"].as<float>(0.5f);
  config.entropy_coef = node["entropy_coef"].as<float>(0.01f);
  config.num_epochs = node["num_epochs"].as<long>(1);
  config.mini_batch_size = node["mini_batch_size"].as<long>(2048);
  config.num_mini_batches = node["num_mini_batches"].as<long>(32);
  config.gae_discount = node["gae_discount"].as<float>(0.99f);
  config.gae_lambda = node["gae_lambda"].as<float>(0.95f);
  config.max_gradient_norm = node["max_gradient_norm"].as<float>(0.5f);
  config.num_rollouts = node["num_rollouts"].as<size_t>(7000);
  config.log_episode_frequency = node["log_episode_frequency"].as<size_t>(10);
  config.num_workers = node["num_workers"].as<size_t>(16);
  config.worker_batch_size = node["worker_batch_size"].as<size_t>(32);
  config.frame_skip = node["frame_skip"].as<size_t>(4);
  config.max_return = node["max_return"].as<float>(-1.0f);
  config.record_observation = node["record_observation"].as<bool>(false);
  return config;
}

template <typename T> float mean(const std::vector<T> &values) {
  if (values.empty())
    throw std::invalid_argument("Values vector is empty.");
  return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

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

void log_data(TensorBoardLogger &logger, const ai::rollout::Log &log,
              const ai::ppo::train::Metrics &metrics, double lr) {
  if (!log.episode_returns.empty()) {
    logger.add_scalar("mean_episode_return", log.steps,
                      mean(log.episode_returns));
    logger.add_scalar("mean_episode_length", log.steps,
                      mean(log.episode_lengths));
    logger.add_histogram("episode_returns", log.steps, log.episode_returns);
    logger.add_histogram("episode_lengths", log.steps, log.episode_lengths);

    if (!log.game_returns.empty()) {
      logger.add_scalar("mean_game_return", log.steps, mean(log.game_returns));
      logger.add_scalar("mean_game_length", log.steps, mean(log.game_lengths));
      logger.add_histogram("game_returns", log.steps, log.game_returns);
      logger.add_histogram("game_lengths", log.steps, log.game_lengths);
    }
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
  if (metrics.clipped_gradients.numel() > 1)
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

  logger.add_scalar("learning_rate", log.steps, lr);
}

torch::nn::Conv2d layer_init(torch::nn::Conv2d layer,
                             double std = std::sqrt(2.0), double bias = 0.0) {
  torch::nn::init::orthogonal_(layer->weight, std);
  if (layer->bias.defined()) {
    torch::nn::init::constant_(layer->bias, bias);
  }
  return layer;
}

torch::nn::Linear layer_init(torch::nn::Linear layer,
                             double std = std::sqrt(2.0), double bias = 0.0) {
  torch::nn::init::orthogonal_(layer->weight, std);
  if (layer->bias.defined()) {
    torch::nn::init::constant_(layer->bias, bias);
  }
  return layer;
}

struct NetworkImpl : torch::nn::Module {
  NetworkImpl(size_t hidden_size, size_t action_size)
      : sequential(layer_init(torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(4, 32, 8).stride(4))),
                   torch::nn::ReLU(),
                   layer_init(torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(32, 64, 4).stride(2))),
                   torch::nn::ReLU(),
                   layer_init(torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(64, 64, 3).stride(1))),
                   torch::nn::ReLU(), torch::nn::Flatten(),
                   layer_init(torch::nn::Linear(64 * 7 * 7, hidden_size))),
        action_head(
            layer_init(torch::nn::Linear(hidden_size, action_size), 0.01)),
        value_head(layer_init(torch::nn::Linear(hidden_size, 1), 1)) {
    register_module("sequential", sequential);
    register_module("action_head", action_head);
    register_module("value_head", value_head);
  }

  struct OutputType {
    torch::Tensor logits;
    torch::Tensor value;
  };

  OutputType forward(torch::Tensor x) {
    {
      torch::NoGradGuard no_grad;
      if (x.device().is_cuda())
        x = x.to(torch::kFloat32);
      if (PREPROCESSED_GRAYSCALE) {
        x = ai::vision::resize_frame_stacked_grayscale_images(x);
      } else {
        // TODO: Investigate why the performance is worse compared to
        //  preprocessed grayscale.
        x = ai::vision::rgb_to_grayscale_frame_stacked_images(x);
        x = ai::vision::resize_frame_stacked_grayscale_images(x);
      }
      x.divide_(255.0);
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

ai::ppo::train::Batch prepare_batch(ai::buffer::Batch &batch) {
  auto observations = batch.observations.flatten(0, 1);
  auto actions = batch.actions.ravel();
  auto advantages = batch.advantages.ravel();
  auto logits = batch.logits.view({-1, batch.logits.size(2)});
  auto returns = batch.returns.ravel();
  auto masks = batch.masks.ravel();
  auto log_probabilities = ai::ppo::losses::normalize_logits(logits);
  ai::ppo::train::Batch other_batch = {observations, actions, log_probabilities,
                                       advantages,   returns, masks};
  return other_batch;
}

ai::ppo::train::Hyperparameters prepare_hyperparameters(const Config &config) {
  ai::ppo::train::Hyperparameters hp = {
      config.clip_param, config.value_loss_coef,
      static_cast<float>(get_annealed_entropy_coef(config.entropy_coef)),
      config.max_gradient_norm};
  return hp;
}

void enable_torch_determinism(uint64_t seed) {
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

int main(int argc, char **argv) {
  const auto start_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  const auto rom_path = std::filesystem::path(argv[1]);
  const auto logger_path = std::filesystem::path(argv[2]).replace_extension(
      "tfevents." + std::to_string(start_time));
  const auto video_path = std::filesystem::path(argv[3]);
  const std::string group_name = argv[4];
  const auto config = load_config(std::filesystem::path(argv[5]));
  std::filesystem::path profile_path;
  if (argc == 7) {
    profile_path = std::filesystem::path(argv[6]);
  }
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

  enable_torch_determinism(42);

  TensorBoardLogger logger(logger_path);
  Network network(config.hidden_size, config.action_size);
  network->to(device);
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(config.learning_rate).eps(1e-5));

  ai::rollout::Rollout rollout(
      rom_path, config.total_environments, config.horizon, config.max_steps,
      config.frame_stack, PREPROCESSED_GRAYSCALE,
      [&network, &device, action_size = config.action_size](
          const torch::Tensor &obs) -> ai::rollout::ActionResult {
        network->eval();
        torch::NoGradGuard no_grad;
        auto observations = device.is_cuda() ? obs.to(torch::kFloat32) : obs;
        auto output = network->forward(observations.to(device));
        auto logits = output.logits;
        auto probabilities = torch::nn::functional::softmax(logits, -1);
        auto actions = torch::multinomial(probabilities, 1, true);
        return {actions.ravel(),
                logits.reshape({-1, static_cast<long>(action_size)}),
                output.value.ravel()};
      },
      config.gae_discount, config.gae_lambda, device, 0, config.num_workers,
      config.worker_batch_size, config.frame_skip, config.max_return,
      video_path);
  torch::Tensor indices =
      torch::empty(config.mini_batch_size * config.num_mini_batches,
                   torch::TensorOptions().dtype(torch::kLong).device(device));
  ai::ppo::train::Metrics metrics(config.num_epochs, config.num_mini_batches,
                                  config.mini_batch_size, device);

  logger.add_hparams(get_parameters(config), group_name, start_time);

  ai::buffer::Batch b;
  {
    torch::NoGradGuard no_grad;
    b = rollout.rollout().batch;
  }
  ai::ppo::train::Batch batch = prepare_batch(b);
  ai::rollout::RolloutResult result;

  at::cuda::CUDAGraph graph;
  network->train();
  if (USE_CUDA_GRAPH) {
    auto hp = prepare_hyperparameters(config);
    ai::ppo::train::capture_train_cuda_graph(graph, network, optimizer, metrics,
                                             indices, batch, config.num_epochs,
                                             config.num_mini_batches, hp, 10);
  }
  if (!profile_path.empty()) {
    torch::autograd::profiler::ProfilerConfig profiler_config =
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::KINETO);
    auto activities = {torch::autograd::profiler::ActivityType::CUDA,
                       torch::autograd::profiler::ActivityType::CPU};
    torch::autograd::profiler::prepareProfiler(profiler_config, activities);
    torch::autograd::profiler::enableProfiler(
        profiler_config, activities,
        {torch::RecordScope::FUNCTION, torch::RecordScope::USER_SCOPE});
  }
  for (size_t rollout_index = 0; rollout_index < config.num_rollouts;
       ++rollout_index) {
    std::cout << "Rollout " << rollout_index + 1 << " of "
              << config.num_rollouts << std::endl;
    auto lr = config.learning_rate *
              (1.0 - rollout_index / static_cast<double>(config.num_rollouts));
    static_cast<torch::optim::AdamOptions &>(
        optimizer.param_groups()[0].options())
        .lr(lr);

    {
      torch::NoGradGuard no_grad;
      result = rollout.rollout();
      auto b = prepare_batch(result.batch);
      batch.copy_(b);
    }
    if (USE_CUDA_GRAPH) {
      ai::ppo::train::train_cuda_graph(graph);
    } else {
      auto hp = prepare_hyperparameters(config);
      ai::ppo::train::train(network, optimizer, metrics, indices, batch,
                            config.num_epochs, config.num_mini_batches, hp);
    }

    log_data(logger, result.log, metrics,
             static_cast<torch::optim::AdamOptions &>(
                 optimizer.param_groups()[0].options())
                 .lr());
    sum += std::accumulate(result.log.episode_returns.begin(),
                           result.log.episode_returns.end(), 0.0f);
    count += result.log.episode_returns.size();
  }
  if (!profile_path.empty()) {
    auto profiler_result = torch::autograd::profiler::disableProfiler();
    profiler_result->save(profile_path);
  }
  std::cout << "Success" << std::endl;
  return 0;
}
