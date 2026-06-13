#include "ai/ppo/train.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <CLI/CLI.hpp>
#include <ale/common/Log.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <type_traits>

#include "ai/checkpoint.h"
#include "ai/ppo/loop.h"
#include "ai/rollout.h"
#include "ai/tensor_util.h"
#include "ai/torch_runtime.h"
#include "stop_signal.h"
#include "tensorboard_logger.h"
#include "training/config.h"
#include "training/session.h"

namespace {

// Seed used when config.deterministic is set (matches the historical value).
constexpr uint64_t kSeed = 42;

std::map<std::string, google::protobuf::Value> get_parameters(
    const training::Config &config, size_t action_size) {
  std::map<std::string, google::protobuf::Value> hparams;
  auto put = [&](const char *name, const auto &field) {
    google::protobuf::Value value;
    if constexpr (std::is_same_v<std::decay_t<decltype(field)>, bool>)
      value.set_bool_value(field);
    else
      value.set_number_value(static_cast<double>(field));
    hparams[name] = value;
  };
  training::for_each_field(config, put);
  // action_size is a property of the ROM (ALE's minimal action set), not a
  // configured value, so it is supplied at runtime rather than from YAML.
  put("action_size", action_size);
  return hparams;
}

void log_data(TensorBoardLogger &logger, const ai::rollout::Log &log,
              const ai::ppo::train::Metrics &metrics, double lr) {
  using ai::tensor_util::gather;
  using ai::tensor_util::mean;
  using ai::tensor_util::to_vector;
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

struct Arguments {
  std::filesystem::path rom_path;
  // Prefix; the timestamp + ".tfevents" suffix is appended in main().
  std::filesystem::path log_path;
  std::filesystem::path profile_path;  // empty when --profile omitted
  std::optional<std::filesystem::path> video_path;  // set only when recording
  std::string group_name;
  training::Config config;
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

  training::Config config = training::load_config(config_path);
  // --video-dir is only consumed when the config asks to record, but a missing
  // path then would silently disable recording -- fail loudly instead.
  if (config.record_video && video_dir.empty()) {
    spdlog::error("record_video is enabled but --video-dir was not provided.");
    std::exit(1);
  }
  try {
    training::validate(config);
  } catch (const std::exception &e) {
    spdlog::error(e.what());
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

}  // namespace

int main(int argc, char **argv) {
  stop_signal::StopSignal stop{SIGTERM, SIGINT};

  // ALE prints a per-environment ROM banner / seed line at Info level straight
  // to stderr (not via spdlog). Quieten it to Warning so the console only shows
  // our logs; genuine ALE warnings/errors still come through. Mode is a
  // process-wide static, so this one call covers every worker's interface too.
  ale::Logger::setMode(ale::Logger::Warning);

  const Arguments args = parse_arguments(argc, argv);
  const training::Config &config = args.config;

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

  const torch::Device device = ai::torch_runtime::select_device();

  if (!std::filesystem::exists(logger_path.parent_path()))
    std::filesystem::create_directories(logger_path.parent_path());
  if (args.video_path.has_value() &&
      !std::filesystem::exists(args.video_path.value()))
    std::filesystem::create_directories(args.video_path.value());

  TensorBoardLogger logger(logger_path);
  training::Session session(config, args.rom_path, args.video_path, device,
                            kSeed);
  logger.add_hparams(get_parameters(config, session.action_size()),
                     args.group_name, start_time);

  if (!args.profile_path.empty()) {
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

  ai::checkpoint::Checkpointer checkpointer(
      run_dir, config.checkpoint_interval,
      [&logger](size_t step, const std::string &text) {
        logger.add_text("checkpoint", step, text.c_str());
      });

  // The training session runs the (collect, update) pipeline internally; this
  // loop only drives it one rollout at a time so it can log and checkpoint
  // each result and bail promptly on SIGINT/SIGTERM.
  for (size_t rollout_index = 0; rollout_index < config.num_rollouts;
       ++rollout_index) {
    if (stop.requested()) {
      spdlog::info("Stop requested — finalizing and shutting down...");
      break;
    }
    spdlog::info("Rollout {} of {}", rollout_index + 1, config.num_rollouts);
    auto report = session.step();
    if (!report) break;
    log_data(logger, *report->log, *report->metrics, report->learning_rate);
    checkpointer.on_rollout_end(report->rollout_index, report->global_step,
                                report->mean_episode_return, session.network(),
                                session.optimizer());
  }

  if (!args.profile_path.empty()) {
    auto profiler_result = torch::autograd::profiler::disableProfiler();
    profiler_result->save(args.profile_path);
  }
  // rollout's video recorder is finalized by RAII as the session unwinds.
  spdlog::info(stop.requested() ? "Interrupted" : "Success");
  return 0;
}
