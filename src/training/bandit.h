#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "training/config.h"

// Bandit-based hyperparameter optimization (Successive Halving, the core of
// Hyperband): treat each hyperparameter configuration as an arm, train arms to
// a small budget, keep the best fraction, and reinvest the saved compute by
// training the survivors longer. Pure orchestration — the actual training is
// injected as an Evaluator, so this stays independent of the training stack.
namespace training::bandit {

// How a single tunable field is drawn.
enum class Distribution {
  kUniform,     // real, uniform over [low, high]
  kLogUniform,  // real, uniform in log-space over [low, high] (needs low > 0) —
                // the natural scale for rates/coefficients spanning decades
  kIntUniform,  // integer, uniform over [low, high] inclusive
  kChoice,      // categorical, uniform over `choices`
};

// The constraint on one Config field the tuner may vary: which field, how to
// draw a value, and how to write that value back into a Config (with the right
// cast). Continuous/integer fields use [low, high]; kChoice enumerates
// `choices`. A field with no ParamSpec keeps its base-config value.
struct ParamSpec {
  std::string name;  // the Config/YAML key this varies (for logging)
  Distribution distribution;
  double low = 0.0;  // bounds for kUniform / kLogUniform / kIntUniform
  double high = 0.0;
  std::vector<double> choices;                  // used by kChoice
  std::function<void(Config &, double)> apply;  // writes the drawn value
};

// The set of fields the tuner may vary, with their constraints.
using SearchSpace = std::vector<ParamSpec>;

// The built-in search space over the standard PPO knobs (learning_rate,
// clip_param, entropy_coef, num_epochs, gae_lambda). All other Config fields
// are held at their base value.
SearchSpace default_search_space();

// Rejects a malformed search space (low > high, non-positive log bounds, empty
// or missing choices, no apply). Throws std::invalid_argument.
void validate(const SearchSpace &space);

// A one-line human-readable description of a spec's constraint, for logging.
std::string describe(const ParamSpec &spec);

// A sampled hyperparameter configuration: the base config with this arm's
// tunable fields overridden. `id` is stable across rungs for logging.
struct Arm {
  size_t id;
  Config config;
};

// Draws `count` arms by sampling each spec in `space` and applying it to a copy
// of `base`. Deterministic given `seed`. Throws if `space` is malformed.
std::vector<Arm> sample_arms(const Config &base, size_t count, uint64_t seed,
                             const SearchSpace &space);

// Trains `arm` to `budget` rollouts and returns its score (higher is better);
// the implementation is expected to set config.num_rollouts = budget. A score
// of -inf means the arm produced no usable signal at this budget.
using Evaluator = std::function<double(const Arm &arm, size_t budget)>;

// One rung: every surviving arm scored at the rung's budget, sorted best-first.
struct RungResult {
  size_t rung;
  size_t budget;
  std::vector<std::pair<Arm, double>> scored;  // descending by score
};

struct Bracket {
  std::vector<RungResult> rungs;
  Arm best;
  double best_score;
};

// Successive Halving: score every arm at `rung_budget`, keep the top 1/eta,
// multiply the budget by eta, and repeat until a single arm remains. Returns
// every rung (the last holds the winner). `evaluate` is called once per
// (arm, budget); restarting from scratch each rung needs no checkpoint resume.
Bracket successive_halving(std::vector<Arm> arms, size_t eta,
                           size_t rung_budget, const Evaluator &evaluate);

}  // namespace training::bandit
