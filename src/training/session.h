#pragma once
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
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

  // One (update, collect) iteration; nullopt once num_rollouts is exhausted.
  // The returned report is valid until the next step() call.
  std::optional<ai::ppo::RolloutReport> step() { return loop_->step(); }

  // Exposed so the caller can checkpoint.
  torch::nn::Module &network() { return *network_; }
  torch::optim::Adam &optimizer() { return *optimizer_; }

  // ALE's minimal action set size for this ROM, for hparam logging.
  size_t action_size() const { return action_size_; }

 private:
  // Publishes the learner's weights to the behaviour policy (async only).
  void sync_actor();
  std::function<ai::rollout::ActionResult(const torch::Tensor &)>
  make_action_selector();

  // Declaration order matters for teardown: rollout_ (whose worker threads run
  // the action selector that reads actor_/device_) must be destroyed before
  // those members, so it is declared after them.
  torch::Device device_;
  bool async_update_;
  size_t action_size_ = 0;
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
