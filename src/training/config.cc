#include "training/config.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace training {

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

void save_config(const Config &config, const std::filesystem::path &path) {
  YAML::Node node;
  for_each_field(
      config, [&](const char *name, const auto &field) { node[name] = field; });
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write config to " + path.string());
  out << node << "\n";
}

void apply_field(Config &config, const std::string &name, double value) {
  bool found = false;
  for_each_field(config, [&](const char *field_name, auto &field) {
    if (found || name != field_name) return;
    using T = std::decay_t<decltype(field)>;
    if constexpr (std::is_same_v<T, bool>)
      field = (value != 0.0);
    else if constexpr (std::is_arithmetic_v<T>)
      field = static_cast<T>(value);
    else
      // Non-numeric fields (e.g. resume_from) are not search-space tunable.
      throw std::invalid_argument("Config field '" + name +
                                  "' cannot be set from a numeric value.");
    found = true;
  });
  if (!found) throw std::invalid_argument("Unknown config field: " + name);
}

void reconcile(Config &config) {
  // mini_batch_size is not free: train() splits the rollout into
  // num_mini_batches contiguous slices (src/ai/ppo/train.h), so it is exactly
  // the rollout size (horizon * total_environments) over that count. Storing it
  // keeps Session's preallocated index/metrics buffers the right shape
  // (src/training/session.cc). A zero count is left for validate() to report.
  if (config.num_mini_batches > 0)
    config.mini_batch_size = static_cast<long>(
        config.horizon * config.total_environments /
        static_cast<size_t>(config.num_mini_batches));
}

void validate(const Config &config) {
  // The learner thread and the rollout draw from the same global RNG, so their
  // interleaving makes runs irreproducible regardless of seeding.
  if (config.async_update && config.deterministic)
    throw std::invalid_argument(
        "async_update is incompatible with deterministic.");

  // Batch geometry is coupled: these mirror the invariants the PPO update
  // (src/ai/ppo/train.h) and the Rollout (src/ai/rollout.cc) enforce at
  // construction, surfaced here so a tuner can reject a bad arm up front instead
  // of crashing deep in setup. mini_batch_size is derived -- see reconcile().
  if (config.num_mini_batches <= 0)
    throw std::invalid_argument("num_mini_batches must be > 0.");
  if (config.mini_batch_size <= 0)
    throw std::invalid_argument("mini_batch_size must be > 0.");
  const size_t size = config.horizon * config.total_environments;
  const auto num_mini_batches = static_cast<size_t>(config.num_mini_batches);
  if (size % num_mini_batches != 0)
    throw std::invalid_argument(
        "num_mini_batches must divide horizon * total_environments.");
  if (static_cast<size_t>(config.mini_batch_size) * num_mini_batches != size)
    throw std::invalid_argument(
        "mini_batch_size must equal horizon * total_environments / "
        "num_mini_batches; call reconcile() after changing the batch shape.");
  if (config.pipeline_groups == 0)
    throw std::invalid_argument("pipeline_groups must be > 0.");
  if (config.total_environments % config.pipeline_groups != 0)
    throw std::invalid_argument(
        "pipeline_groups must divide total_environments.");
  if (config.worker_batch_size == 0)
    throw std::invalid_argument("worker_batch_size must be > 0.");
  if ((config.total_environments / config.pipeline_groups) %
          config.worker_batch_size !=
      0)
    throw std::invalid_argument(
        "worker_batch_size must divide the pipeline group size "
        "(total_environments / pipeline_groups).");
}

}  // namespace training
