#include "ai/ppo/loop.h"

#include <future>
#include <utility>

#include "ai/ppo/async_updater.h"
#include "ai/tensor_util.h"

namespace ai::ppo {

Loop::Loop(size_t num_rollouts, double learning_rate, Trainer &trainer,
           ai::rollout::Rollout &rollout, ai::ppo::train::Metrics &metrics,
           AsyncUpdater *updater, std::function<void()> publish_weights,
           size_t start_rollout_index)
    : num_rollouts_(num_rollouts),
      learning_rate_(learning_rate),
      trainer_(trainer),
      rollout_(rollout),
      metrics_(metrics),
      updater_(updater),
      publish_weights_(std::move(publish_weights)),
      rollout_index_(start_rollout_index) {
  if (rollout_index_ < num_rollouts_) {
    torch::NoGradGuard no_grad;
    current_ = rollout_.rollout();
  }
}

std::optional<RolloutReport> Loop::step() {
  if (rollout_index_ >= num_rollouts_) return std::nullopt;
  // Promote the rollout collected during the previous step (which overlapped
  // that step's update) into the one we now train on. The first step trains on
  // the rollout collected in the constructor.
  if (!first_step_) current_ = std::move(next_);
  first_step_ = false;
  const size_t k = rollout_index_;

  trainer_.set_learning_rate(learning_rate_ *
                             (1.0 - k / static_cast<double>(num_rollouts_)));

  std::future<void> update_done;
  auto update_job = [&] { trainer_.update(current_.batch); };
  if (updater_)
    update_done = updater_->submit(update_job);
  else
    update_job();

  // Collect rollout k+1 with the actor's current weights, overlapping the
  // update of rollout k when running asynchronously.
  if (k + 1 < num_rollouts_) {
    torch::NoGradGuard no_grad;
    next_ = rollout_.rollout();
  }
  // Joining the updater includes its GPU work, so the learner's weights are
  // final before they are published to the actor and reported.
  if (update_done.valid()) update_done.get();
  publish_weights_();

  RolloutReport report;
  report.rollout_index = k;
  report.global_step = current_.log.steps;
  report.mean_episode_return =
      current_.log.episode_returns.empty()
          ? std::nullopt
          : std::optional<double>(
                ai::tensor_util::mean(current_.log.episode_returns));
  report.log = &current_.log;
  report.metrics = &metrics_;
  report.learning_rate = trainer_.learning_rate();

  ++rollout_index_;
  return report;
}

}  // namespace ai::ppo
