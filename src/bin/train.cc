#include "ai/ppo/train.h"
#include "ai/ppo/losses.h"
#include "ai/rollout.h"
#include "ai/vision.h"
#include "stop_signal.h"
#include "tensorboard_logger.h"
#include <ale/ale_interface.hpp>
#include <ale/common/Log.hpp>
#include <ale/version.hpp>
#include <limits>
#include <numeric>
#include <spdlog/spdlog.h>
#include <torch/nn.h>
#include <type_traits>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

struct Config {
  size_t total_environments;
  size_t hidden_size;
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
  size_t num_workers;
  size_t worker_batch_size;
  size_t frame_skip;
  // Some games like breakout have a maximum return
  // which should be used to reset the environment.
  float max_return;
  // It is faster to record using the observation.
  // However the observation may be in grayscale.
  bool record_observation;
  bool record_video;
  bool cuda_graph;
  bool deterministic;
  // Write latest.pt every `checkpoint_interval` rollouts (0 disables all
  // checkpointing) and best.pt whenever mean episode return improves, both into
  // a run directory keyed by the same start_time stamp as the tfevents file.
  size_t checkpoint_interval;
  // Path to a checkpoint .pt to restore network + optimizer + step from; empty
  // starts fresh.
  std::string resume_from;
};

// The single place where YAML keys bind to Config members. Each field keeps its
// real type, so the loader and logger below stay strongly typed. action_size is
// excluded: it is constant and not loaded from YAML (logged separately).
template <typename Self, typename Visitor>
void for_each_field(Self &config, Visitor &&visit) {
  visit("total_environments", config.total_environments);
  visit("hidden_size", config.hidden_size);
  visit("horizon", config.horizon);
  visit("max_steps", config.max_steps);
  visit("frame_stack", config.frame_stack);
  visit("learning_rate", config.learning_rate);
  visit("clip_param", config.clip_param);
  visit("value_loss_coef", config.value_loss_coef);
  visit("entropy_coef", config.entropy_coef);
  visit("num_epochs", config.num_epochs);
  visit("mini_batch_size", config.mini_batch_size);
  visit("num_mini_batches", config.num_mini_batches);
  visit("gae_discount", config.gae_discount);
  visit("gae_lambda", config.gae_lambda);
  visit("max_gradient_norm", config.max_gradient_norm);
  visit("num_rollouts", config.num_rollouts);
  visit("num_workers", config.num_workers);
  visit("worker_batch_size", config.worker_batch_size);
  visit("frame_skip", config.frame_skip);
  visit("max_return", config.max_return);
  visit("record_observation", config.record_observation);
  visit("record_video", config.record_video);
  visit("cuda_graph", config.cuda_graph);
  visit("deterministic", config.deterministic);
  visit("checkpoint_interval", config.checkpoint_interval);
  visit("resume_from", config.resume_from);
}

std::map<std::string, google::protobuf::Value>
get_parameters(const Config &config, size_t action_size) {
  std::map<std::string, google::protobuf::Value> hparams;
  auto put = [&](const char *name, const auto &field) {
    using Field = std::decay_t<decltype(field)>;
    google::protobuf::Value value;
    if constexpr (std::is_same_v<Field, bool>)
      value.set_bool_value(field);
    else if constexpr (std::is_same_v<Field, std::string>)
      value.set_string_value(field);
    else
      value.set_number_value(static_cast<double>(field));
    hparams[name] = value;
  };
  for_each_field(config, put);
  // action_size is a property of the ROM (ALE's minimal action set), not a
  // configured value, so it is supplied at runtime rather than from YAML.
  put("action_size", action_size);
  return hparams;
}

Config load_config(const std::filesystem::path &path) {
  Config config;
  YAML::Node node = YAML::LoadFile(path.string());
  for_each_field(config, [&](const char *name, auto &field) {
    // Every key is required: a missing key is a hard error rather than a
    // silent default.
    if (!node[name])
      throw std::runtime_error(std::string("Missing config key: ") + name);
    field = node[name].as<std::decay_t<decltype(field)>>();
  });
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

// `step` is the absolute global env step (log.steps offset by any resumed run's
// starting step) so a resumed run's curves continue rather than restart at 0.
void log_data(TensorBoardLogger &logger, const ai::rollout::Log &log,
              const ai::ppo::train::Metrics &metrics, double lr, size_t step) {
  if (!log.episode_returns.empty()) {
    logger.add_scalar("mean_episode_return", step, mean(log.episode_returns));
    logger.add_scalar("mean_episode_length", step, mean(log.episode_lengths));
    logger.add_histogram("episode_returns", step, log.episode_returns);
    logger.add_histogram("episode_lengths", step, log.episode_lengths);

    if (!log.game_returns.empty()) {
      logger.add_scalar("mean_game_return", step, mean(log.game_returns));
      logger.add_scalar("mean_game_length", step, mean(log.game_lengths));
      logger.add_histogram("game_returns", step, log.game_returns);
      logger.add_histogram("game_lengths", step, log.game_lengths);
    }
  }
  logger.add_scalar("mean_clipped_gradient", step,
                    metrics.clipped_gradients.mean().item<float>());
  logger.add_scalar("mean_loss", step, metrics.loss.mean().item<float>());
  logger.add_scalar("mean_clipped_loss", step,
                    mean(metrics.clipped_losses, metrics.masks));
  logger.add_scalar("mean_value_loss", step,
                    mean(metrics.value_losses, metrics.masks));
  logger.add_scalar("mean_entropy", step,
                    mean(metrics.entropies, metrics.masks));
  logger.add_scalar("mean_ratio", step, mean(metrics.ratio, metrics.masks));
  if (metrics.clipped_gradients.numel() > 1)
    logger.add_histogram("clipped_gradients", step,
                         to_vector(metrics.clipped_gradients));
  logger.add_histogram("losses", step,
                       gather(metrics.total_losses, metrics.masks));
  logger.add_histogram("clipped_losses", step,
                       gather(metrics.clipped_losses, metrics.masks));
  logger.add_histogram("value_losses", step,
                       gather(metrics.value_losses, metrics.masks));
  logger.add_histogram("entropies", step,
                       gather(metrics.entropies, metrics.masks));
  logger.add_histogram("ratios", step, gather(metrics.ratio, metrics.masks));
  logger.add_histogram("advantages", step,
                       gather(metrics.advantages, metrics.masks));
  logger.add_histogram("returns", step,
                       gather(metrics.returns, metrics.masks));

  logger.add_scalar("learning_rate", step, lr);
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
      x = x.to(torch::kFloat32);
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
  ai::ppo::train::Hyperparameters hp = {config.clip_param,
                                        config.value_loss_coef,
                                        config.entropy_coef,
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

// Everything needed to resume a run: model weights, optimizer moments, the next
// rollout to run (so the LR schedule continues), and the best return seen so the
// best.pt criterion survives a resume.
struct Checkpoint {
  size_t next_rollout_index;
  double best_return;
  // Absolute global env step at save time, so a resumed run's TensorBoard
  // curves continue from here instead of restarting at 0.
  size_t global_step;
};

void save_checkpoint(const std::filesystem::path &path, const Network &network,
                     const torch::optim::Adam &optimizer,
                     const Checkpoint &state) {
  torch::serialize::OutputArchive archive;
  torch::serialize::OutputArchive model_archive;
  network->save(model_archive);
  archive.write("model", model_archive);
  torch::serialize::OutputArchive optimizer_archive;
  optimizer.save(optimizer_archive);
  archive.write("optimizer", optimizer_archive);
  archive.write("next_rollout_index",
                c10::IValue(static_cast<int64_t>(state.next_rollout_index)));
  archive.write("best_return", c10::IValue(state.best_return));
  archive.write("global_step",
                c10::IValue(static_cast<int64_t>(state.global_step)));
  // Write to a sibling temp file then rename: rename is atomic on a single
  // filesystem, so a crash mid-write can never truncate an existing checkpoint.
  auto tmp = path;
  tmp += ".tmp";
  archive.save_to(tmp.string());
  std::filesystem::rename(tmp, path);
}

Checkpoint load_checkpoint(const std::filesystem::path &path, Network &network,
                           torch::optim::Adam &optimizer,
                           const torch::Device &device) {
  torch::serialize::InputArchive archive;
  // Remap storages onto the current device so a CPU-saved checkpoint resumes on
  // GPU and vice versa. Module::load copies into the existing parameters in
  // place, keeping the tensors the optimizer already references valid.
  archive.load_from(path.string(), device);
  torch::serialize::InputArchive model_archive;
  archive.read("model", model_archive);
  network->load(model_archive);
  torch::serialize::InputArchive optimizer_archive;
  archive.read("optimizer", optimizer_archive);
  optimizer.load(optimizer_archive);
  c10::IValue next_rollout_index, best_return, global_step;
  archive.read("next_rollout_index", next_rollout_index);
  archive.read("best_return", best_return);
  archive.read("global_step", global_step);
  return {static_cast<size_t>(next_rollout_index.toInt()),
          best_return.toDouble(), static_cast<size_t>(global_step.toInt())};
}

int main(int argc, char **argv) {
  stop_signal::StopSignal stop{SIGTERM, SIGINT};

  // ALE prints a per-environment ROM banner / seed line at Info level straight
  // to stderr (not via spdlog). Quieten it to Warning so the console only shows
  // our logs; genuine ALE warnings/errors still come through. Mode is a
  // process-wide static, so this one call covers every worker's interface too.
  ale::Logger::setMode(ale::Logger::Warning);
  const auto start_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  const auto rom_path = std::filesystem::path(argv[1]);
  const auto config = load_config(std::filesystem::path(argv[5]));
  // A run is a self-contained directory holding its event file and checkpoints.
  // A fresh run gets a new directory keyed by start_time. Resuming reuses the
  // directory that holds the checkpoint, so TensorBoard merges the new event
  // file into the same run and (with the restored global step) the metric
  // curves continue rather than restarting.
  const std::filesystem::path run_dir =
      config.resume_from.empty()
          ? std::filesystem::path(argv[2]).parent_path() /
                (std::filesystem::path(argv[2]).filename().string() + "." +
                 std::to_string(start_time))
          : std::filesystem::path(config.resume_from).parent_path();
  const std::filesystem::path logger_path =
      run_dir / (std::filesystem::path(argv[2]).filename().string() +
                 ".tfevents." + std::to_string(start_time));
  const std::optional<std::filesystem::path> video_path =
      config.record_video
          ? std::optional<std::filesystem::path>(std::filesystem::path(argv[3]))
          : std::nullopt;
  const std::string group_name = argv[4];
  std::filesystem::path profile_path;
  if (argc == 7) {
    profile_path = std::filesystem::path(argv[6]);
  }
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

  if (!std::filesystem::exists(logger_path.parent_path())) {
    std::filesystem::create_directories(logger_path.parent_path());
  }
  if (video_path.has_value() && !std::filesystem::exists(video_path.value())) {
    std::filesystem::create_directories(video_path.value());
  }

  if (config.deterministic)
    enable_torch_determinism(42);

  // The action count is a property of the ROM, not a configured value: read it
  // from ALE's minimal action set (matching what the rollout buffer uses).
  size_t action_size;
  {
    ale::ALEInterface ale;
    ale.loadROM(rom_path);
    action_size = ale.getMinimalActionSet().size();
  }

  TensorBoardLogger logger(logger_path);
  Network network(config.hidden_size, action_size);
  network->to(device);
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(config.learning_rate).eps(1e-5));

  // Resume before the initial rollout and any CUDA-graph capture so the graph
  // captures the restored weights/optimizer state. The rollout RNG and env
  // state are not saved, so resumption is approximate, not bit-exact.
  size_t start_rollout_index = 0;
  double best_return = -std::numeric_limits<double>::infinity();
  size_t step_offset = 0;
  if (!config.resume_from.empty()) {
    const Checkpoint state =
        load_checkpoint(config.resume_from, network, optimizer, device);
    start_rollout_index = state.next_rollout_index;
    best_return = state.best_return;
    step_offset = state.global_step;
    spdlog::info("Resumed from {} at rollout {} (global step {})",
                 config.resume_from, start_rollout_index, step_offset);
  }

  ai::rollout::Rollout rollout(
      rom_path, config.total_environments, config.horizon, config.max_steps,
      config.frame_stack, true,
      [&network, &device, action_size](
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
      video_path, config.record_observation);
  torch::Tensor indices =
      torch::empty(config.mini_batch_size * config.num_mini_batches,
                   torch::TensorOptions().dtype(torch::kLong).device(device));
  ai::ppo::train::Metrics metrics(config.num_epochs, config.num_mini_batches,
                                  config.mini_batch_size, device);

  logger.add_hparams(get_parameters(config, action_size), group_name,
                     start_time);

  ai::buffer::Batch b;
  {
    torch::NoGradGuard no_grad;
    b = rollout.rollout().batch;
  }
  ai::ppo::train::Batch batch = prepare_batch(b);
  ai::rollout::RolloutResult result;

#ifdef __linux__
  at::cuda::CUDAGraph graph;
  network->train();
  if (config.cuda_graph) {
    auto hp = prepare_hyperparameters(config);
    ai::ppo::train::capture_train_cuda_graph(graph, network, optimizer, metrics,
                                             indices, batch, config.num_epochs,
                                             config.num_mini_batches, hp, 10);
  }
#endif
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
  for (size_t rollout_index = start_rollout_index;
       rollout_index < config.num_rollouts; ++rollout_index) {
    if (stop.requested()) {
      spdlog::info("Stop requested — finalizing and shutting down...");
      break;
    }
    spdlog::info("Rollout {} of {}", rollout_index + 1, config.num_rollouts);
    auto lr = config.learning_rate *
              (1.0 - rollout_index / static_cast<double>(config.num_rollouts));
    static_cast<torch::optim::AdamOptions &>(
        optimizer.param_groups()[0].options())
        .lr(lr);

    {
      torch::NoGradGuard no_grad;
      result = rollout.rollout();
    }
    if (config.cuda_graph) {
#ifdef __linux__
      auto b = prepare_batch(result.batch);
      batch.copy_(b);
      ai::ppo::train::train_cuda_graph(graph);
#else
      TORCH_CHECK(false, "cuda_graph is only supported on Linux (__linux__ not "
                         "defined). Set cuda_graph=false or run on Linux.");
#endif

    } else {
      batch = prepare_batch(result.batch);
      auto hp = prepare_hyperparameters(config);
      ai::ppo::train::train(network, optimizer, metrics, indices, batch,
                            config.num_epochs, config.num_mini_batches, hp);
    }

    // Absolute global step: continues across a resume so the curves don't
    // restart at 0.
    const size_t step = step_offset + result.log.steps;
    log_data(logger, result.log, metrics,
             static_cast<torch::optim::AdamOptions &>(
                 optimizer.param_groups()[0].options())
                 .lr(),
             step);

    if (config.checkpoint_interval > 0) {
      // next_rollout_index is rollout_index + 1: resuming continues with the
      // rollout after the one just trained on.
      if (!result.log.episode_returns.empty()) {
        const double rollout_return = mean(result.log.episode_returns);
        if (rollout_return > best_return) {
          best_return = rollout_return;
          save_checkpoint(run_dir / "best.pt", network, optimizer,
                          {rollout_index + 1, best_return, step});
          logger.add_text("checkpoint", step,
                          ("best.pt return=" + std::to_string(best_return) +
                           " rollout=" + std::to_string(rollout_index + 1))
                              .c_str());
        }
      }
      if ((rollout_index + 1) % config.checkpoint_interval == 0) {
        save_checkpoint(run_dir / "latest.pt", network, optimizer,
                        {rollout_index + 1, best_return, step});
        logger.add_text(
            "checkpoint", step,
            ("latest.pt rollout=" + std::to_string(rollout_index + 1)).c_str());
      }
    }
  }
  if (!profile_path.empty()) {
    auto profiler_result = torch::autograd::profiler::disableProfiler();
    profiler_result->save(profile_path);
  }
  // rollout's video recorder is finalized by RAII as this scope unwinds.
  spdlog::info(stop.requested() ? "Interrupted" : "Success");
  return 0;
}
