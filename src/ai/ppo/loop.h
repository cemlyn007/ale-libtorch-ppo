#pragma once
#include <torch/torch.h>

#include <cstddef>
#include <functional>
#include <optional>

#include "ai/ppo/train.h"
#include "ai/ppo/trainer.h"
#include "ai/rollout.h"

namespace ai::ppo {

class AsyncUpdater;

// Per-rollout summary handed back after each step(). The `log` and `metrics`
// pointers reference state owned by the Loop and stay valid only until the next
// step() call, so callers must consume them before stepping again.
struct RolloutReport {
  size_t rollout_index;
  size_t global_step;
  std::optional<double> mean_episode_return;
  const ai::rollout::Log *log;
  const ai::ppo::train::Metrics *metrics;
  double learning_rate;
};

// The PPO training driver: a software pipeline of depth one over
// (collect, update). It knows nothing about the concrete network or config —
// only the Trainer interface, the Rollout, the shared Metrics buffer, and a
// callback to publish freshly-trained weights to the behaviour policy.
class Loop {
 public:
  // num_rollouts/learning_rate drive the linear LR anneal. `updater` may be
  // null for the synchronous path (the update runs inline); when non-null each
  // update is submitted to it so it overlaps the next collection.
  // `publish_weights` runs after each update is joined (republish the actor
  // snapshot in async mode; a no-op otherwise). The loop runs rollouts
  // [start_rollout_index, num_rollouts); start_rollout_index > 0 resumes a run,
  // continuing the LR anneal from that index. Collects the first rollout up
  // front when there is one to run.
  Loop(size_t num_rollouts, double learning_rate, Trainer &trainer,
       ai::rollout::Rollout &rollout, ai::ppo::train::Metrics &metrics,
       AsyncUpdater *updater, std::function<void()> publish_weights,
       size_t start_rollout_index = 0);

  // Run one (update, collect) iteration and return the report for the rollout
  // just trained on, or nullopt once num_rollouts iterations are exhausted.
  std::optional<RolloutReport> step();

 private:
  size_t num_rollouts_;
  double learning_rate_;
  Trainer &trainer_;
  ai::rollout::Rollout &rollout_;
  ai::ppo::train::Metrics &metrics_;
  AsyncUpdater *updater_;
  std::function<void()> publish_weights_;
  size_t rollout_index_;
  // True until the first step() consumes the constructor-collected rollout;
  // distinguishes it from the next_ promoted on later steps, so a resumed run
  // starting at rollout_index_ > 0 still trains current_ first.
  bool first_step_ = true;
  // current_ is the rollout under training; next_ is collected during a step
  // (overlapping that step's update) and promoted to current_ at the next step.
  ai::rollout::RolloutResult current_;
  ai::rollout::RolloutResult next_;
};

}  // namespace ai::ppo
