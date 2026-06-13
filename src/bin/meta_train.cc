#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <CLI/CLI.hpp>
#include <ale/common/Log.hpp>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <type_traits>

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

}  // namespace

int main(int argc, char **argv) {
  stop_signal::StopSignal stop{SIGTERM, SIGINT};
  ale::Logger::setMode(ale::Logger::Warning);

  CLI::App app{"Bandit (Successive Halving) hyperparameter search for PPO."};
  std::filesystem::path rom_path, base_config_path, log_path;
  size_t num_arms = 9, eta = 3, rung_budget = 20;
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
  app.add_option("--arms", num_arms, "Initial arms (rung-0 population).")
      ->capture_default_str();
  app.add_option("--eta", eta, "Halving factor: keep the top 1/eta each rung.")
      ->capture_default_str();
  app.add_option("--rung-budget", rung_budget,
                 "Rollouts trained at rung 0; multiplied by eta each rung.")
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
      training::bandit::default_search_space();
  spdlog::info("Search space:");
  for (const auto &spec : space)
    spdlog::info("    {}", training::bandit::describe(spec));

  auto arms = training::bandit::sample_arms(base, num_arms, seed, space);
  spdlog::info("Sampled {} arms; eta={}, rung-0 budget={} rollouts.", num_arms,
               eta, rung_budget);

  // Train one arm to `budget` rollouts and return its best mean episode return.
  // Video and checkpointing are inherently off here: Session is handed no video
  // path and does not checkpoint, so the search leaves no per-arm artifacts.
  auto evaluate = [&](const training::bandit::Arm &arm,
                      size_t budget) -> double {
    if (stop.requested()) return -std::numeric_limits<double>::infinity();
    training::Config config = arm.config;
    config.num_rollouts = budget;

    const auto ts = std::chrono::system_clock::now().time_since_epoch().count();
    const std::string name =
        "arm" + std::to_string(arm.id) + "_b" + std::to_string(budget);
    const std::filesystem::path run_dir =
        log_path / (name + "." + std::to_string(ts));
    std::filesystem::create_directories(run_dir);
    TensorBoardLogger logger(
        (run_dir / (name + ".tfevents." + std::to_string(ts))).string());

    training::Session session(config, rom_path, std::nullopt, device,
                              seed + arm.id);
    logger.add_hparams(get_parameters(config, session.action_size()), name, ts);

    double best = -std::numeric_limits<double>::infinity();
    for (auto report = session.step(); report; report = session.step()) {
      logger.add_scalar("mean_loss", report->global_step,
                        report->metrics->loss.mean().item<float>());
      if (report->mean_episode_return) {
        logger.add_scalar("mean_episode_return", report->global_step,
                          *report->mean_episode_return);
        best = std::max(best, *report->mean_episode_return);
      }
      if (stop.requested()) break;
    }
    spdlog::info(
        "  arm {:>3}  budget {:>6}  lr={:.2e} clip={:.2f} ent={:.2e} "
        "epochs={} lambda={:.3f}  -> best_return={:.4f}",
        arm.id, budget, config.learning_rate, config.clip_param,
        config.entropy_coef, config.num_epochs, config.gae_lambda, best);
    return best;
  };

  auto bracket = training::bandit::successive_halving(std::move(arms), eta,
                                                      rung_budget, evaluate);

  spdlog::info("=== Successive Halving bracket ===");
  for (const auto &rung : bracket.rungs) {
    spdlog::info("Rung {} (budget {}): {} arms", rung.rung, rung.budget,
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
