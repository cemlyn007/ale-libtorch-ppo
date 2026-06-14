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
#include "ai/torch_runtime.h"
#include "stop_signal.h"
#include "tensorboard_logger.h"
#include "training/config.h"
#include "training/logging.h"
#include "training/session.h"

namespace {

// Seed used when config.deterministic is set (matches the historical value).
constexpr uint64_t kSeed = 42;

std::map<std::string, google::protobuf::Value> get_parameters(
    const training::Config &config, size_t action_size) {
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
  training::for_each_field(config, put);
  // action_size is a property of the ROM (ALE's minimal action set), not a
  // configured value, so it is supplied at runtime rather than from YAML.
  put("action_size", action_size);
  return hparams;
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
  // event file and its checkpoints. Resuming reuses the checkpoint's directory
  // so the resumed process writes its event file into the same run, which
  // TensorBoard merges into one continuous timeline.
  const std::filesystem::path log_path = args.log_path;
  const std::filesystem::path run_dir =
      config.resume_from.empty()
          ? log_path.parent_path() / (log_path.filename().string() + "." +
                                      std::to_string(start_time))
          : std::filesystem::path(config.resume_from).parent_path();
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

  // Seed best_return so best.pt is only rewritten on a genuine improvement over
  // the resumed run, not on the first rollout after a resume.
  ai::checkpoint::Checkpointer checkpointer(
      run_dir, config.checkpoint_interval,
      [&logger](size_t step, const std::string &text) {
        logger.add_text("checkpoint", step, text.c_str());
      },
      session.best_return());

  // The training session runs the (collect, update) pipeline internally; this
  // loop only drives it one rollout at a time so it can log and checkpoint each
  // result. SIGINT/SIGTERM is checked between rollouts, so a stop takes effect
  // once the in-flight step() returns — which still collects the next rollout,
  // since the generic Loop has no view of the stop signal.
  for (size_t rollout_index = session.start_rollout_index();
       rollout_index < config.num_rollouts; ++rollout_index) {
    if (stop.requested()) {
      spdlog::info("Stop requested — finalizing and shutting down...");
      break;
    }
    spdlog::info("Rollout {} of {}", rollout_index + 1, config.num_rollouts);
    auto report = session.step();
    if (!report) break;
    // Absolute global step: continues across a resume so the curves don't
    // restart at 0.
    const size_t step = session.step_offset() + report->global_step;
    training::log_rollout(logger, *report->log, *report->metrics,
                          report->learning_rate, step);
    checkpointer.on_rollout_end(report->rollout_index, step,
                                report->mean_episode_return, session.network(),
                                session.optimizer());
  }
  // A graceful stop breaks before the next interval, and a clean run can finish
  // between intervals; either way flush the last completed rollout so latest.pt
  // holds the newest weights rather than only the last interval multiple.
  checkpointer.flush_latest(session.network(), session.optimizer(),
                            stop.requested() ? "shutdown" : "final");

  if (!args.profile_path.empty()) {
    auto profiler_result = torch::autograd::profiler::disableProfiler();
    profiler_result->save(args.profile_path);
  }
  // rollout's video recorder is finalized by RAII as the session unwinds.
  spdlog::info(stop.requested() ? "Interrupted" : "Success");
  return 0;
}
