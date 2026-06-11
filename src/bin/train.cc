#include "ai/ppo/train.h"

#include <spdlog/spdlog.h>
#include <torch/nn.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <CLI/CLI.hpp>
#include <ale/ale_interface.hpp>
#include <ale/common/Log.hpp>
#include <ale/version.hpp>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>

#include "ai/ppo/losses.h"
#include "ai/rollout.h"
#include "ai/vision.h"
#include "checkpoint.h"
#include "stop_signal.h"
#include "tensorboard_logger.h"

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
  // checkpointing) and best.pt whenever mean episode return improves, into a
  // run directory keyed by the same start_time stamp as the tfevents file.
  size_t checkpoint_interval;
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
}

std::map<std::string, google::protobuf::Value> get_parameters(
    const Config &config, size_t action_size) {
  std::map<std::string, google::protobuf::Value> hparams;
  auto put = [&](const char *name, const auto &field) {
    google::protobuf::Value value;
    if constexpr (std::is_same_v<std::decay_t<decltype(field)>, bool>)
      value.set_bool_value(field);
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

template <typename T>
float mean(const std::vector<T> &values) {
  if (values.empty()) throw std::invalid_argument("Values vector is empty.");
  return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

std::vector<float> to_vector(const torch::Tensor &tensor) {
  auto t = tensor.contiguous().to(torch::kCPU, torch::kFloat);
  float *data_ptr = t.data_ptr<float>();
  return std::vector<float>(data_ptr, data_ptr + t.numel());
}

std::vector<float> gather(const torch::Tensor &tensor,
                          const torch::Tensor &mask) {
  return to_vector(tensor.masked_select(mask));
}

void log_data(TensorBoardLogger &logger, const ai::rollout::Log &log,
              const ai::ppo::train::Metrics &metrics, double lr) {
  const auto step = log.steps;
  const auto &masks = metrics.masks;
  auto scalar = [&](const char *tag, double v) {
    logger.add_scalar(tag, step, v);
  };
  auto hist = [&](const char *tag, const auto &v) {
    logger.add_histogram(tag, step, v);
  };
  auto g = [&](const torch::Tensor &t) { return gather(t, masks); };
  // Gather once on the host, then log both the mean and the distribution from
  // the same vector — avoids a second masked_select + device sync per tensor.
  auto scalar_and_hist = [&](const char *mean_tag, const char *hist_tag,
                             const torch::Tensor &t) {
    auto v = g(t);
    scalar(mean_tag, mean(v));
    hist(hist_tag, v);
  };

  if (!log.episode_returns.empty()) {
    scalar("mean_episode_return", mean(log.episode_returns));
    scalar("mean_episode_length", mean(log.episode_lengths));
    hist("episode_returns", log.episode_returns);
    hist("episode_lengths", log.episode_lengths);
    if (!log.game_returns.empty()) {
      scalar("mean_game_return", mean(log.game_returns));
      scalar("mean_game_length", mean(log.game_lengths));
      hist("game_returns", log.game_returns);
      hist("game_lengths", log.game_lengths);
    }
  }

  scalar("mean_clipped_gradient",
         metrics.clipped_gradients.mean().item<float>());
  scalar("mean_loss", metrics.loss.mean().item<float>());
  scalar_and_hist("mean_clipped_loss", "clipped_losses",
                  metrics.clipped_losses);
  scalar_and_hist("mean_value_loss", "value_losses", metrics.value_losses);
  scalar_and_hist("mean_entropy", "entropies", metrics.entropies);
  scalar_and_hist("mean_ratio", "ratios", metrics.ratio);

  if (metrics.clipped_gradients.numel() > 1)
    hist("clipped_gradients", to_vector(metrics.clipped_gradients));
  hist("losses", g(metrics.total_losses));
  hist("advantages", g(metrics.advantages));
  hist("returns", g(metrics.returns));

  scalar("learning_rate", lr);
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
  ai::ppo::train::Hyperparameters hp = {
      config.clip_param, config.value_loss_coef, config.entropy_coef,
      config.max_gradient_norm};
  return hp;
}

// Read/write the learning rate on the optimizer's single Adam param group.
double get_optimizer_lr(torch::optim::Optimizer &optimizer) {
  return static_cast<torch::optim::AdamOptions &>(
             optimizer.param_groups()[0].options())
      .lr();
}
void set_optimizer_lr(torch::optim::Optimizer &optimizer, double lr) {
  static_cast<torch::optim::AdamOptions &>(
      optimizer.param_groups()[0].options())
      .lr(lr);
}

// Strategy for turning a fresh rollout into one optimization pass. The two
// implementations differ only in how the network is driven: the eager path
// re-runs the autograd training loop each call; the CUDA-graph path captures
// that loop once and replays it, refreshing its inputs in place.
struct Trainer {
  virtual ~Trainer() = default;
  // Schedule the next learning rate. Honoured eagerly; a no-op once a CUDA
  // graph has baked the rate in at capture time (see make_trainer).
  virtual void set_learning_rate(double lr) = 0;
  // The learning rate actually in effect, for honest logging.
  virtual double learning_rate() = 0;
  // Refresh inputs from the latest rollout and run one optimization pass,
  // writing per-update metrics into the shared Metrics buffer.
  virtual void update(ai::buffer::Batch &rollout) = 0;
};

class EagerTrainer : public Trainer {
 public:
  EagerTrainer(Network network, torch::optim::Optimizer &optimizer,
               ai::ppo::train::Metrics &metrics, torch::Tensor &indices,
               size_t num_epochs, size_t num_mini_batches,
               ai::ppo::train::Hyperparameters hyperparameters)
      : network_(std::move(network)),
        optimizer_(optimizer),
        metrics_(metrics),
        indices_(indices),
        num_epochs_(num_epochs),
        num_mini_batches_(num_mini_batches),
        hyperparameters_(hyperparameters) {}

  void set_learning_rate(double lr) override {
    set_optimizer_lr(optimizer_, lr);
  }
  double learning_rate() override { return get_optimizer_lr(optimizer_); }

  void update(ai::buffer::Batch &rollout) override {
    auto batch = prepare_batch(rollout);
    ai::ppo::train::train(network_, optimizer_, metrics_, indices_, batch,
                          num_epochs_, num_mini_batches_, hyperparameters_);
  }

 private:
  Network network_;
  torch::optim::Optimizer &optimizer_;
  ai::ppo::train::Metrics &metrics_;
  torch::Tensor &indices_;
  size_t num_epochs_;
  size_t num_mini_batches_;
  ai::ppo::train::Hyperparameters hyperparameters_;
};

#ifdef __linux__
class CudaGraphTrainer : public Trainer {
 public:
  CudaGraphTrainer(Network network, torch::optim::Optimizer &optimizer,
                   ai::ppo::train::Metrics &metrics, torch::Tensor &indices,
                   size_t num_epochs, size_t num_mini_batches,
                   ai::ppo::train::Hyperparameters hyperparameters,
                   ai::ppo::train::Batch batch)
      : network_(std::move(network)),
        optimizer_(optimizer),
        batch_(std::move(batch)),
        hyperparameters_(hyperparameters) {
    network_->train();
    ai::ppo::train::capture_train_cuda_graph(
        graph_, network_, optimizer_, metrics, indices, batch_, num_epochs,
        num_mini_batches, hyperparameters_, 10);
    // Capture bakes the optimizer's lr into the graph as a host scalar; replays
    // cannot observe later changes, so annealing is silently disabled.
    spdlog::warn(
        "CUDA graph enabled: learning rate frozen at {} for the run "
        "(annealing disabled).",
        get_optimizer_lr(optimizer_));
  }

  void set_learning_rate(double) override {}  // baked in at capture; no-op
  double learning_rate() override { return get_optimizer_lr(optimizer_); }

  void update(ai::buffer::Batch &rollout) override {
    auto batch = prepare_batch(rollout);
    batch_.copy_(batch);
    ai::ppo::train::train_cuda_graph(graph_);
  }

 private:
  Network network_;
  torch::optim::Optimizer &optimizer_;
  ai::ppo::train::Batch batch_;
  ai::ppo::train::Hyperparameters hyperparameters_;
  at::cuda::CUDAGraph graph_;
};
#endif

// Builds the training strategy once, up front: the CUDA-graph path captures its
// graph here (in the constructor) and unsupported platforms are rejected
// eagerly rather than failing inside the rollout loop. initial_batch is the
// persistent input buffer for capture; the eager path ignores it.
std::unique_ptr<Trainer> make_trainer(const Config &config, Network network,
                                      torch::optim::Optimizer &optimizer,
                                      ai::ppo::train::Metrics &metrics,
                                      torch::Tensor &indices,
                                      ai::ppo::train::Batch initial_batch) {
  auto hyperparameters = prepare_hyperparameters(config);
  if (!config.cuda_graph)
    return std::make_unique<EagerTrainer>(
        std::move(network), optimizer, metrics, indices, config.num_epochs,
        config.num_mini_batches, hyperparameters);
#ifdef __linux__
  return std::make_unique<CudaGraphTrainer>(
      std::move(network), optimizer, metrics, indices, config.num_epochs,
      config.num_mini_batches, hyperparameters, std::move(initial_batch));
#else
  (void)initial_batch;
  throw std::runtime_error(
      "cuda_graph is only supported on Linux; set cuda_graph=false.");
#endif
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

struct Arguments {
  std::filesystem::path rom_path;
  // Prefix; the timestamp + ".tfevents" suffix is appended in main().
  std::filesystem::path log_path;
  std::filesystem::path profile_path;  // empty when --profile omitted
  std::optional<std::filesystem::path> video_path;  // set only when recording
  std::string group_name;
  Config config;
};

Arguments parse_arguments(int argc, char **argv) {
  CLI::App app{"Train a PPO agent on an Atari ROM."};
  std::filesystem::path rom_path, log_path, config_path, video_dir,
      profile_path;
  std::string group_name;
  app.add_option("--rom", rom_path, "Atari ROM to train on.")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("--config", config_path, "YAML config file.")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("--log-path", log_path,
                 "TensorBoard log path prefix; '.tfevents.<timestamp>' is "
                 "appended at runtime.")
      ->required();
  app.add_option("--group", group_name,
                 "Group name for hyperparameters logged to TensorBoard.")
      ->required();
  app.add_option("--video-dir", video_dir,
                 "Directory to write videos to. Required when record_video is "
                 "set in the config.");
  app.add_option("--profile", profile_path,
                 "Path to write a libtorch (Perfetto) profile to.");
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  Config config = load_config(config_path);
  // --video-dir is only consumed when the config asks to record, but a missing
  // path then would silently disable recording -- fail loudly instead.
  if (config.record_video && video_dir.empty()) {
    spdlog::error("record_video is enabled but --video-dir was not provided.");
    std::exit(1);
  }
  std::optional<std::filesystem::path> video_path =
      config.record_video
          ? std::optional<std::filesystem::path>(std::move(video_dir))
          : std::nullopt;

  return Arguments{std::move(rom_path),     std::move(log_path),
                   std::move(profile_path), std::move(video_path),
                   std::move(group_name),   std::move(config)};
}

int main(int argc, char **argv) {
  stop_signal::StopSignal stop{SIGTERM, SIGINT};

  // ALE prints a per-environment ROM banner / seed line at Info level straight
  // to stderr (not via spdlog). Quieten it to Warning so the console only shows
  // our logs; genuine ALE warnings/errors still come through. Mode is a
  // process-wide static, so this one call covers every worker's interface too.
  ale::Logger::setMode(ale::Logger::Warning);

  const Arguments args = parse_arguments(argc, argv);
  const Config &config = args.config;
  const std::filesystem::path &rom_path = args.rom_path;
  const std::optional<std::filesystem::path> &video_path = args.video_path;
  const std::filesystem::path &profile_path = args.profile_path;
  const std::string &group_name = args.group_name;

  const auto start_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  // A run is a self-contained directory <log_path>.<start_time>/ holding its
  // event file and its checkpoints, so each run's artifacts stay together.
  const std::filesystem::path log_path = args.log_path;
  const std::filesystem::path run_dir =
      log_path.parent_path() /
      (log_path.filename().string() + "." + std::to_string(start_time));
  const std::filesystem::path logger_path =
      run_dir / (log_path.filename().string() + ".tfevents." +
                 std::to_string(start_time));

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

  if (config.deterministic) enable_torch_determinism(42);

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

  ai::rollout::Rollout rollout(
      rom_path, config.total_environments, config.horizon, config.max_steps,
      config.frame_stack, true,
      [&network, &device,
       action_size](const torch::Tensor &obs) -> ai::rollout::ActionResult {
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

  ai::ppo::train::Batch initial_batch;
  {
    torch::NoGradGuard no_grad;
    auto b = rollout.rollout().batch;
    initial_batch = prepare_batch(b);
  }
  // On the CUDA-graph path make_trainer captures the graph here, using
  // initial_batch as its persistent input buffer; the eager path ignores it.
  std::unique_ptr<Trainer> trainer = make_trainer(
      config, network, optimizer, metrics, indices, std::move(initial_batch));
  ai::rollout::RolloutResult result;
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
  // Mean episode return of the best rollout so far; best.pt is rewritten
  // whenever a rollout beats it.
  double best_return = -std::numeric_limits<double>::infinity();
  for (size_t rollout_index = 0; rollout_index < config.num_rollouts;
       ++rollout_index) {
    if (stop.requested()) {
      spdlog::info("Stop requested — finalizing and shutting down...");
      break;
    }
    spdlog::info("Rollout {} of {}", rollout_index + 1, config.num_rollouts);
    trainer->set_learning_rate(
        config.learning_rate *
        (1.0 - rollout_index / static_cast<double>(config.num_rollouts)));

    {
      torch::NoGradGuard no_grad;
      result = rollout.rollout();
    }
    trainer->update(result.batch);

    log_data(logger, result.log, metrics, trainer->learning_rate());

    if (config.checkpoint_interval > 0) {
      // next_rollout_index is rollout_index + 1 and global_step is the env-step
      // count so far: both are written so a checkpoint is resume-ready, even
      // though loading them back to resume a run is a later change.
      const size_t step = result.log.steps;
      if (!result.log.episode_returns.empty()) {
        const double rollout_return = mean(result.log.episode_returns);
        if (rollout_return > best_return) {
          best_return = rollout_return;
          checkpoint::save(run_dir / "best.pt", *network, optimizer,
                           {rollout_index + 1, best_return, step});
          logger.add_text("checkpoint", step,
                          ("best.pt return=" + std::to_string(best_return) +
                           " rollout=" + std::to_string(rollout_index + 1))
                              .c_str());
        }
      }
      if ((rollout_index + 1) % config.checkpoint_interval == 0) {
        checkpoint::save(run_dir / "latest.pt", *network, optimizer,
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
