#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <CLI/CLI.hpp>
#include <ale/common/Log.hpp>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "ai/ppo/loop.h"
#include "ai/ppo/train.h"
#include "ai/torch_runtime.h"
#include "stop_signal.h"
#include "tensorboard_logger.h"
#include "training/bandit.h"
#include "training/config.h"
#include "training/session.h"

namespace {

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
  put("action_size", action_size);
  return hparams;
}

// Robust upper estimate of a learning curve: the maximum of its rolling mean
// over a window of `w` rollouts. Rewards a sustained high plateau rather than a
// single lucky spike. With fewer than `w` points, falls back to the mean of all
// of them; an empty curve scores -inf (no usable signal in the budget).
double smoothed_curve_max(const std::vector<double> &curve, size_t w) {
  if (curve.empty()) return -std::numeric_limits<double>::infinity();
  w = std::max<size_t>(1, w);
  if (curve.size() <= w)
    return std::accumulate(curve.begin(), curve.end(), 0.0) / curve.size();
  double window_sum = std::accumulate(curve.begin(), curve.begin() + w, 0.0);
  double best = window_sum / w;
  for (size_t i = w; i < curve.size(); ++i) {
    window_sum += curve[i] - curve[i - w];
    best = std::max(best, window_sum / w);
  }
  return best;
}

}  // namespace

int main(int argc, char **argv) {
  stop_signal::StopSignal stop{SIGTERM, SIGINT};
  ale::Logger::setMode(ale::Logger::Warning);

  CLI::App app{"Bandit (Successive Halving) hyperparameter search for PPO."};
  std::filesystem::path rom_path, base_config_path, log_path, search_space_path;
  size_t num_arms = 9, eta = 3, smoothing_window = 5;
  double rung_seconds = 60.0;
  uint64_t seed = 0;
  app.add_option("--rom", rom_path, "Atari ROM to train on.")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("--base-config", base_config_path,
                 "YAML config the arms are sampled around.")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("--log-path", log_path,
                 "Directory for per-arm TensorBoard logs and the winning "
                 "config (best_config.yaml).")
      ->required();
  app.add_option("--search-space", search_space_path,
                 "YAML search space (see configs/search_space.yaml); uses the "
                 "built-in space if omitted.")
      ->check(CLI::ExistingFile);
  app.add_option("--arms", num_arms, "Initial arms (rung-0 population).")
      ->capture_default_str();
  app.add_option("--eta", eta, "Halving factor: keep the top 1/eta each rung.")
      ->capture_default_str();
  app.add_option("--rung-seconds", rung_seconds,
                 "Wall-clock training seconds per arm at rung 0; multiplied by "
                 "eta each rung.")
      ->capture_default_str();
  app.add_option("--smoothing-window", smoothing_window,
                 "Rollout window for the rolling-mean used to score each arm.")
      ->capture_default_str();
  app.add_option("--seed", seed, "Seed for arm sampling and per-arm training.")
      ->capture_default_str();
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }
  if (eta < 2) {
    spdlog::error("--eta must be >= 2.");
    return 1;
  }

  training::Config base = training::load_config(base_config_path);
  training::validate(base);

  const torch::Device device = ai::torch_runtime::select_device();
  std::filesystem::create_directories(log_path);

  const training::bandit::SearchSpace space =
      search_space_path.empty()
          ? training::bandit::default_search_space()
          : training::bandit::load_search_space(search_space_path);
  spdlog::info("Search space ({}):", search_space_path.empty()
                                         ? "built-in"
                                         : search_space_path.string());
  for (const auto &spec : space)
    spdlog::info("    {}", training::bandit::describe(spec));

  auto arms = training::bandit::sample_arms(base, num_arms, seed, space);
  spdlog::info(
      "Sampled {} arms; eta={}, rung-0 budget={}s, smoothing window={}.",
      num_arms, eta, rung_seconds, smoothing_window);

  // Train one arm for `budget` wall-clock seconds and score it by the robust
  // peak of its return curve (max of the rolling mean). Video and checkpointing
  // are inherently off: Session is handed no video path and does not
  // checkpoint, so the search leaves no per-arm artifacts.
  auto evaluate = [&](const training::bandit::Arm &arm,
                      double budget_seconds) -> double {
    if (stop.requested()) return -std::numeric_limits<double>::infinity();
    training::Config config = arm.config;
    // Time-bounded, not rollout-bounded: lift the rollout cap so wall-clock is
    // the only limiter. This also holds the learning rate ~constant within a
    // probe (the anneal is a fraction of num_rollouts), so arms are compared
    // under a fixed rate rather than each rung's own anneal schedule.
    config.num_rollouts = std::numeric_limits<size_t>::max();

    // An arm whose sampled config is internally inconsistent (e.g. a batch
    // geometry the rollout/update reject) scores -inf and is culled rather than
    // taking the whole search down. validate() catches it up front; the try also
    // covers anything Session construction throws.
    try {
      training::validate(config);

      const auto ts =
          std::chrono::system_clock::now().time_since_epoch().count();
      const std::string name =
          "arm" + std::to_string(arm.id) + "_t" +
          std::to_string(static_cast<long>(budget_seconds));
      const std::filesystem::path run_dir =
          log_path / (name + "." + std::to_string(ts));
      std::filesystem::create_directories(run_dir);
      TensorBoardLogger logger(
          (run_dir / (name + ".tfevents." + std::to_string(ts))).string());

      training::Session session(config, rom_path, std::nullopt, device,
                                seed + arm.id);
      logger.add_hparams(get_parameters(arm.config, session.action_size()),
                         name, ts);

      // Wall-clock measured over training only (Session construction — env
      // startup, graph capture — is fixed per-arm overhead, excluded here).
      std::vector<double> curve;
      size_t rollouts = 0;
      const auto start = std::chrono::steady_clock::now();
      for (auto report = session.step(); report; report = session.step()) {
        ++rollouts;
        logger.add_scalar("mean_loss", report->global_step,
                          report->metrics->loss.mean().item<float>());
        if (report->mean_episode_return) {
          logger.add_scalar("mean_episode_return", report->global_step,
                            *report->mean_episode_return);
          curve.push_back(*report->mean_episode_return);
        }
        const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (elapsed >= budget_seconds || stop.requested()) break;
      }
      const double score = smoothed_curve_max(curve, smoothing_window);
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      spdlog::info(
          "  arm {:>3}  {:>6.1f}s ({:>4} rollouts)  lr={:.2e} clip={:.2f} "
          "ent={:.2e} epochs={} lambda={:.3f}  -> score={:.4f}",
          arm.id, elapsed, rollouts, config.learning_rate, config.clip_param,
          config.entropy_coef, config.num_epochs, config.gae_lambda, score);
      return score;
    } catch (const std::exception &e) {
      spdlog::warn("  arm {:>3}  rejected: {}", arm.id, e.what());
      return -std::numeric_limits<double>::infinity();
    }
  };

  auto bracket = training::bandit::successive_halving(std::move(arms), eta,
                                                      rung_seconds, evaluate);

  spdlog::info("=== Successive Halving bracket ===");
  for (const auto &rung : bracket.rungs) {
    spdlog::info("Rung {} (budget {:.1f}s): {} arms", rung.rung, rung.budget,
                 rung.scored.size());
    for (const auto &[arm, score] : rung.scored)
      spdlog::info("    arm {:>3}  score={:.4f}", arm.id, score);
  }
  const auto &best = bracket.best.config;
  spdlog::info(
      "BEST: arm {} score={:.4f}  lr={:.3e} clip={:.2f} ent={:.3e} epochs={} "
      "lambda={:.3f}",
      bracket.best.id, bracket.best_score, best.learning_rate, best.clip_param,
      best.entropy_coef, best.num_epochs, best.gae_lambda);

  const std::filesystem::path best_path = log_path / "best_config.yaml";
  training::save_config(bracket.best.config, best_path);
  spdlog::info("Wrote winning config to {}", best_path.string());
  return 0;
}
