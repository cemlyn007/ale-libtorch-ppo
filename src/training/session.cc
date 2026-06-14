#include "training/session.h"

#include <ale/ale_interface.hpp>
#include <spdlog/spdlog.h>
#include <utility>

#include "ai/checkpoint.h"
#include "ai/torch_runtime.h"
#include "training/trainer.h"

namespace training {

Session::Session(const Config &config, std::filesystem::path rom_path,
                 std::optional<std::filesystem::path> video_path,
                 const torch::Device &device, uint64_t seed)
    : device_(device), async_update_(config.async_update) {
  if (config.deterministic) ai::torch_runtime::enable_determinism(seed);

  // The action count is a property of the ROM, not a configured value: read it
  // from ALE's minimal action set (matching what the rollout buffer uses).
  {
    ale::ALEInterface ale;
    ale.loadROM(rom_path);
    action_size_ = ale.getMinimalActionSet().size();
  }

  network_ = Network(config.hidden_size, action_size_);
  network_->to(device_);
  optimizer_ = std::make_unique<torch::optim::Adam>(
      network_->parameters(),
      torch::optim::AdamOptions(config.learning_rate).eps(1e-5));

  // Restore before the actor snapshot and any CUDA-graph capture so both see
  // the resumed weights/optimizer state. The rollout RNG and env state are not
  // saved, so resumption is approximate, not bit-exact.
  if (!config.resume_from.empty()) {
    const ai::checkpoint::Checkpoint state = ai::checkpoint::load(
        config.resume_from, *network_, *optimizer_, device_);
    start_rollout_index_ = state.next_rollout_index;
    step_offset_ = state.global_step;
    best_return_ = state.best_return;
    spdlog::info("Resumed from {} at rollout {} (global step {})",
                 config.resume_from, start_rollout_index_, step_offset_);
  }

  // Synchronous mode shares the learner's module as the behaviour policy; async
  // mode acts on a snapshot copy so the learner thread can write the real
  // weights while the rollout reads these (sync_actor publishes).
  actor_ = network_;
  if (config.async_update) {
    actor_ = Network(config.hidden_size, action_size_);
    actor_->to(device_);
  }
  sync_actor();

  rollout_ = std::make_unique<ai::rollout::Rollout>(
      rom_path, config.total_environments, config.horizon, config.max_steps,
      config.frame_stack, true, make_action_selector(), config.gae_discount,
      config.gae_lambda, device_, 0, config.num_workers,
      config.worker_batch_size, config.frame_skip, config.max_return,
      std::move(video_path), config.record_observation, config.pipeline_groups,
      // Double buffering keeps the batch under training intact while the next
      // rollout is collected into the other buffer.
      config.async_update ? 2 : 1);

  indices_ =
      torch::empty(config.mini_batch_size * config.num_mini_batches,
                   torch::TensorOptions().dtype(torch::kLong).device(device_));
  metrics_.emplace(config.num_epochs, config.num_mini_batches,
                   config.mini_batch_size, device_);

  ai::ppo::train::Batch initial_batch;
  {
    torch::NoGradGuard no_grad;
    auto b = rollout_->rollout().batch;
    initial_batch = prepare_batch(b);
    // The CUDA-graph trainer keeps initial_batch as its persistent input and
    // copies every fresh rollout into it, but prepare_batch returns views of
    // the rollout buffer: give the graph its own storage so those copies never
    // write into (or race) a buffer a collection is filling.
    if (config.cuda_graph)
      initial_batch = {initial_batch.observations.clone(),
                       initial_batch.actions.clone(),
                       initial_batch.log_probabilities.clone(),
                       initial_batch.advantages.clone(),
                       initial_batch.returns.clone(),
                       initial_batch.masks.clone()};
  }
  // On the CUDA-graph path make_trainer captures the graph here, using
  // initial_batch as its persistent input buffer; the eager path ignores it.
  trainer_ = make_trainer(config, network_, *optimizer_, *metrics_, indices_,
                          std::move(initial_batch));
  // CUDA-graph capture warm-up steps the real weights; republish to the actor.
  sync_actor();

  if (config.async_update)
    updater_ = std::make_unique<ai::ppo::AsyncUpdater>(device_);

  loop_ = std::make_unique<ai::ppo::Loop>(
      config.num_rollouts, config.learning_rate, *trainer_, *rollout_,
      *metrics_, updater_.get(), [this] { sync_actor(); },
      start_rollout_index_);
}

void Session::sync_actor() {
  if (!async_update_) return;
  torch::NoGradGuard no_grad;
  const auto source = network_->parameters();
  auto destination = actor_->parameters();
  for (size_t i = 0; i < source.size(); ++i) destination[i].copy_(source[i]);
}

std::function<ai::rollout::ActionResult(const torch::Tensor &)>
Session::make_action_selector() {
  return [this](const torch::Tensor &obs) -> ai::rollout::ActionResult {
    actor_->eval();
    torch::NoGradGuard no_grad;
    auto observations = device_.is_cuda() ? obs.to(torch::kFloat32) : obs;
    auto output = actor_->forward(observations.to(device_));
    auto logits = output.logits;
    auto probabilities = torch::nn::functional::softmax(logits, -1);
    auto actions = torch::multinomial(probabilities, 1, true);
    return {actions.ravel(),
            logits.reshape({-1, static_cast<long>(action_size_)}),
            output.value.ravel()};
  };
}

}  // namespace training
