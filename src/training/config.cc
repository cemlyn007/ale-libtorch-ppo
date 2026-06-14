#include "training/config.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace training {

Config compose_config(const std::filesystem::path &training_path,
                      const std::filesystem::path &environment_path) {
  // Two flat YAML files bound into one Config: the environment file carries a
  // game's truncation and frame settings, the training file the rest. A key in
  // both resolves to the environment's, so per-game values override a training
  // default. Every schema key must appear in at least one file; one missing
  // from both is a hard error rather than a silent default.
  const YAML::Node environment = YAML::LoadFile(environment_path.string());
  const YAML::Node training = YAML::LoadFile(training_path.string());
  Config config;
  for_each_field(config, [&](const char *name, auto &field) {
    const YAML::Node env_value = environment[name];
    const YAML::Node value = env_value ? env_value : training[name];
    if (!value)
      throw std::runtime_error(std::string("Missing config key: ") + name);
    field = value.as<std::decay_t<decltype(field)>>();
  });
  return config;
}

void validate(const Config &config) {
  // The learner thread and the rollout draw from the same global RNG, so their
  // interleaving makes runs irreproducible regardless of seeding.
  if (config.async_update && config.deterministic)
    throw std::invalid_argument(
        "async_update is incompatible with deterministic.");
}

}  // namespace training
