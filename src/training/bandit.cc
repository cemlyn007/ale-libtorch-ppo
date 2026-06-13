#include "training/bandit.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace training::bandit {

namespace {

// Draw one value for a spec in the field's natural units. Validation (positive
// log bounds, non-empty choices, ordered bounds) is assumed already done.
double sample_value(const ParamSpec &spec, std::mt19937_64 &rng) {
  switch (spec.distribution) {
    case Distribution::kUniform:
      return std::uniform_real_distribution<double>(spec.low, spec.high)(rng);
    case Distribution::kLogUniform:
      return std::exp(std::uniform_real_distribution<double>(
          std::log(spec.low), std::log(spec.high))(rng));
    case Distribution::kIntUniform:
      return static_cast<double>(std::uniform_int_distribution<long>(
          static_cast<long>(spec.low), static_cast<long>(spec.high))(rng));
    case Distribution::kChoice:
      return spec.choices[std::uniform_int_distribution<size_t>(
          0, spec.choices.size() - 1)(rng)];
  }
  throw std::invalid_argument("Unknown distribution for " + spec.name);
}

}  // namespace

SearchSpace default_search_space() {
  return {
      {"learning_rate",
       Distribution::kLogUniform,
       1e-4,
       1e-3,
       {},
       [](Config &c, double v) { c.learning_rate = v; }},
      {"clip_param",
       Distribution::kChoice,
       0.0,
       0.0,
       {0.1, 0.2, 0.3},
       [](Config &c, double v) { c.clip_param = static_cast<float>(v); }},
      {"entropy_coef",
       Distribution::kLogUniform,
       1e-3,
       3e-2,
       {},
       [](Config &c, double v) { c.entropy_coef = static_cast<float>(v); }},
      {"num_epochs",
       Distribution::kIntUniform,
       1,
       4,
       {},
       [](Config &c, double v) { c.num_epochs = static_cast<long>(v); }},
      {"gae_lambda",
       Distribution::kUniform,
       0.90,
       0.98,
       {},
       [](Config &c, double v) { c.gae_lambda = static_cast<float>(v); }},
  };
}

SearchSpace load_search_space(const std::filesystem::path &path) {
  static const std::unordered_map<std::string, Distribution> kDistributions = {
      {"uniform", Distribution::kUniform},
      {"log_uniform", Distribution::kLogUniform},
      {"int_uniform", Distribution::kIntUniform},
      {"choice", Distribution::kChoice},
  };
  const YAML::Node root = YAML::LoadFile(path.string());
  const YAML::Node parameters = root["parameters"];
  if (!parameters || !parameters.IsSequence())
    throw std::runtime_error("search space '" + path.string() +
                             "' must have a top-level 'parameters' sequence.");

  SearchSpace space;
  for (const YAML::Node &entry : parameters) {
    ParamSpec spec;
    spec.name = entry["field"].as<std::string>();
    const std::string distribution = entry["distribution"].as<std::string>();
    const auto it = kDistributions.find(distribution);
    if (it == kDistributions.end())
      throw std::runtime_error("Unknown distribution '" + distribution +
                               "' for field '" + spec.name + "'.");
    spec.distribution = it->second;
    if (spec.distribution == Distribution::kChoice) {
      spec.choices = entry["choices"].as<std::vector<double>>();
    } else {
      spec.low = entry["low"].as<double>();
      spec.high = entry["high"].as<double>();
    }
    // Bind the setter by name through the Config reflection, so a YAML spec can
    // target any field without per-field code here.
    const std::string name = spec.name;
    spec.apply = [name](Config &config, double value) {
      apply_field(config, name, value);
    };
    space.push_back(std::move(spec));
  }
  validate(space);
  return space;
}

void validate(const SearchSpace &space) {
  for (const ParamSpec &spec : space) {
    if (!spec.apply)
      throw std::invalid_argument("ParamSpec '" + spec.name +
                                  "' has no apply function.");
    switch (spec.distribution) {
      case Distribution::kChoice:
        if (spec.choices.empty())
          throw std::invalid_argument("ParamSpec '" + spec.name +
                                      "' is kChoice but has no choices.");
        break;
      case Distribution::kLogUniform:
        if (spec.low <= 0.0)
          throw std::invalid_argument("ParamSpec '" + spec.name +
                                      "' is kLogUniform but low <= 0.");
        [[fallthrough]];
      case Distribution::kUniform:
      case Distribution::kIntUniform:
        if (spec.low > spec.high)
          throw std::invalid_argument("ParamSpec '" + spec.name +
                                      "' has low > high.");
        break;
    }
  }
}

std::string describe(const ParamSpec &spec) {
  switch (spec.distribution) {
    case Distribution::kUniform:
      return spec.name + " ~ uniform[" + std::to_string(spec.low) + ", " +
             std::to_string(spec.high) + "]";
    case Distribution::kLogUniform:
      return spec.name + " ~ log-uniform[" + std::to_string(spec.low) + ", " +
             std::to_string(spec.high) + "]";
    case Distribution::kIntUniform:
      return spec.name + " ~ int[" +
             std::to_string(static_cast<long>(spec.low)) + ", " +
             std::to_string(static_cast<long>(spec.high)) + "]";
    case Distribution::kChoice: {
      std::string s = spec.name + " ~ choice{";
      for (size_t i = 0; i < spec.choices.size(); ++i)
        s += (i ? ", " : "") + std::to_string(spec.choices[i]);
      return s + "}";
    }
  }
  return spec.name;
}

std::vector<Arm> sample_arms(const Config &base, size_t count, uint64_t seed,
                             const SearchSpace &space) {
  validate(space);
  std::mt19937_64 rng(seed);
  std::vector<Arm> arms;
  arms.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    Config config = base;
    for (const ParamSpec &spec : space)
      spec.apply(config, sample_value(spec, rng));
    arms.push_back({i, config});
  }
  return arms;
}

Bracket successive_halving(std::vector<Arm> arms, size_t eta,
                           size_t rung_budget, const Evaluator &evaluate) {
  Bracket bracket;
  std::vector<Arm> alive = std::move(arms);
  size_t rung = 0;
  size_t budget = rung_budget;
  while (true) {
    RungResult result;
    result.rung = rung;
    result.budget = budget;
    for (const Arm &arm : alive)
      result.scored.emplace_back(arm, evaluate(arm, budget));
    std::sort(result.scored.begin(), result.scored.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    bracket.rungs.push_back(std::move(result));

    if (alive.size() <= 1) break;
    // Keep the top 1/eta (at least one) and reinvest the budget in them.
    const size_t keep = std::max<size_t>(1, alive.size() / eta);
    const auto &scored = bracket.rungs.back().scored;
    std::vector<Arm> survivors;
    survivors.reserve(keep);
    for (size_t i = 0; i < keep; ++i) survivors.push_back(scored[i].first);
    alive = std::move(survivors);
    budget *= eta;
    ++rung;
  }
  const auto &winner = bracket.rungs.back().scored.front();
  bracket.best = winner.first;
  bracket.best_score = winner.second;
  return bracket;
}

}  // namespace training::bandit
