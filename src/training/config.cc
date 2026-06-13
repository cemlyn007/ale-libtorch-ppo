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

void validate(const Config &config) {
  // The learner thread and the rollout draw from the same global RNG, so their
  // interleaving makes runs irreproducible regardless of seeding.
  if (config.async_update && config.deterministic)
    throw std::invalid_argument(
        "async_update is incompatible with deterministic.");
}

}  // namespace training
