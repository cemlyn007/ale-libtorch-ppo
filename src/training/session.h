#pragma once
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#include "ai/ppo/async_updater.h"
#include "ai/ppo/loop.h"
#include "ai/ppo/train.h"
#include "ai/ppo/trainer.h"
#include "ai/rollout.h"
#include "training/config.h"
#include "training/network.h"

namespace training {

// One PPO training run, fully wired: owns the network, optimizer, rollout,
// trainer and the generic ai::ppo::Loop, binding this Config + Network onto it.
// The loop machinery lives in //src/ai and never names these concrete types, so
// a tuner can drive many Sessions back to back without touching that library.
class Session {
 public:
  // device is chosen by the caller (ai::torch_runtime::select_device);
  // video_path is set only when recording. seed seeds torch when
  // config.deterministic is set.
  Session(const Config &config, std::filesystem::path rom_path,
          std::optional<std::filesystem::path> video_path,
          const torch::Device &device, uint64_t seed);

  // Non-movable and non-copyable: the action selector captures `this`, so a
  // moved-from Session would leave the rollout calling into a dead object. A
  // tuner driving many runs must hold them by pointer (e.g. a
  // std::vector<std::unique_ptr<Session>>), not by value.
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  // One (update, collect) iteration; nullopt once num_rollouts is exhausted.
  // The returned report is valid until the next step() call.
  std::optional<ai::ppo::RolloutReport> step() { return loop_->step(); }

  // Exposed so the caller can checkpoint.
  torch::nn::Module &network() { return *network_; }
  torch::optim::Adam &optimizer() { return *optimizer_; }

  // ALE's minimal action set size for this ROM, for hparam logging.
  size_t action_size() const { return action_size_; }

  // Resume state restored from config.resume_from (defaults for a fresh run):
  // the rollout index to resume at, the global-step offset that continues the
  // TensorBoard timeline, and the best return so best.pt survives a resume.
  size_t start_rollout_index() const { return start_rollout_index_; }
  size_t step_offset() const { return step_offset_; }
  double best_return() const { return best_return_; }

 private:
  // Publishes the learner's weights to the behaviour policy (async only).
  void sync_actor();
  std::function<ai::rollout::ActionResult(const torch::Tensor &)>
  make_action_selector();

  // Declaration order governs teardown (members destroyed in reverse). rollout_
  // must be torn down before actor_/network_/device_/action_size_: the action
  // selector captures `this` and reads them — it runs on the main thread inside
  // Rollout::rollout(), and ~Rollout also joins the env-stepping workers, so
  // both must finish while those members are still alive. loop_ holds the
  // current_/next_ batches that alias rollout_'s buffers, so it is declared
  // last to release them first. Hence rollout_/loop_ come after the state they
  // use.
  torch::Device device_;
  bool async_update_;
  size_t action_size_ = 0;
  // Resume state; defaults mean "fresh run" when config.resume_from is empty.
  size_t start_rollout_index_ = 0;
  size_t step_offset_ = 0;
  double best_return_ = -std::numeric_limits<double>::infinity();
  Network network_{nullptr};
  Network actor_{nullptr};
  std::unique_ptr<torch::optim::Adam> optimizer_;
  std::optional<ai::ppo::train::Metrics> metrics_;
  torch::Tensor indices_;
  std::unique_ptr<ai::ppo::Trainer> trainer_;
  std::unique_ptr<ai::rollout::Rollout> rollout_;
  std::unique_ptr<ai::ppo::AsyncUpdater> updater_;
  std::unique_ptr<ai::ppo::Loop> loop_;
};

}  // namespace training
