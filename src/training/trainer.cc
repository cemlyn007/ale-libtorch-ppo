#include "training/trainer.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

#include "ai/ppo/losses.h"

namespace training {

ai::ppo::train::Batch prepare_batch(ai::buffer::Batch &batch) {
  auto observations = batch.observations.flatten(0, 1);
  auto actions = batch.actions.ravel();
  auto advantages = batch.advantages.ravel();
  auto logits = batch.logits.view({-1, batch.logits.size(2)});
  auto returns = batch.returns.ravel();
  auto masks = batch.masks.ravel();
  auto log_probabilities = ai::ppo::losses::normalize_logits(logits);
  return {observations, actions, log_probabilities, advantages, returns, masks};
}

namespace {

ai::ppo::train::Hyperparameters prepare_hyperparameters(const Config &config) {
  return {config.clip_param, config.value_loss_coef, config.entropy_coef,
          config.max_gradient_norm, config.shuffle_mini_batches};
}

// Read/write the learning rate on the optimizer's single Adam param group.
double get_optimizer_lr(torch::optim::Optimizer &optimizer) {
  return static_cast<torch::optim::AdamOptions &>(
             optimizer.param_groups()[0].options())
      .lr();
}
void set_optimizer_lr(torch::optim::Optimizer &optimizer, double lr) {
  static_cast<torch::optim::AdamOptions &>(
      optimizer.param_groups()[0].options())
      .lr(lr);
}

// Re-runs the autograd training loop each update.
class EagerTrainer : public ai::ppo::Trainer {
 public:
  EagerTrainer(Network network, torch::optim::Optimizer &optimizer,
               ai::ppo::train::Metrics &metrics, torch::Tensor &indices,
               size_t num_epochs, size_t num_mini_batches,
               ai::ppo::train::Hyperparameters hyperparameters)
      : network_(std::move(network)),
        optimizer_(optimizer),
        metrics_(metrics),
        indices_(indices),
        num_epochs_(num_epochs),
        num_mini_batches_(num_mini_batches),
        hyperparameters_(hyperparameters) {}

  void set_learning_rate(double lr) override {
    set_optimizer_lr(optimizer_, lr);
  }
  double learning_rate() override { return get_optimizer_lr(optimizer_); }

  void update(ai::buffer::Batch &rollout) override {
    auto batch = prepare_batch(rollout);
    ai::ppo::train::train(network_, optimizer_, metrics_, indices_, batch,
                          num_epochs_, num_mini_batches_, hyperparameters_);
  }

 private:
  Network network_;
  torch::optim::Optimizer &optimizer_;
  ai::ppo::train::Metrics &metrics_;
  torch::Tensor &indices_;
  size_t num_epochs_;
  size_t num_mini_batches_;
  ai::ppo::train::Hyperparameters hyperparameters_;
};

#ifdef __linux__
// Captures the autograd training loop once and replays it, refreshing its
// inputs in place.
class CudaGraphTrainer : public ai::ppo::Trainer {
 public:
  CudaGraphTrainer(Network network, torch::optim::Optimizer &optimizer,
                   ai::ppo::train::Metrics &metrics, torch::Tensor &indices,
                   size_t num_epochs, size_t num_mini_batches,
                   ai::ppo::train::Hyperparameters hyperparameters,
                   ai::ppo::train::Batch batch)
      : network_(std::move(network)),
        optimizer_(optimizer),
        batch_(std::move(batch)),
        hyperparameters_(hyperparameters) {
    network_->train();
    ai::ppo::train::capture_train_cuda_graph(
        graph_, network_, optimizer_, metrics, indices, batch_, num_epochs,
        num_mini_batches, hyperparameters_, 10);
    // Capture bakes the optimizer's lr into the graph as a host scalar; replays
    // cannot observe later changes, so annealing is silently disabled.
    spdlog::warn(
        "CUDA graph enabled: learning rate frozen at {} for the run "
        "(annealing disabled).",
        get_optimizer_lr(optimizer_));
  }

  void set_learning_rate(double) override {}  // baked in at capture; no-op
  double learning_rate() override { return get_optimizer_lr(optimizer_); }

  void update(ai::buffer::Batch &rollout) override {
    auto batch = prepare_batch(rollout);
    batch_.copy_(batch);
    ai::ppo::train::train_cuda_graph(graph_);
  }

 private:
  Network network_;
  torch::optim::Optimizer &optimizer_;
  ai::ppo::train::Batch batch_;
  ai::ppo::train::Hyperparameters hyperparameters_;
  at::cuda::CUDAGraph graph_;
};
#endif

}  // namespace

std::unique_ptr<ai::ppo::Trainer> make_trainer(
    const Config &config, Network network, torch::optim::Optimizer &optimizer,
    ai::ppo::train::Metrics &metrics, torch::Tensor &indices,
    ai::ppo::train::Batch initial_batch) {
  auto hyperparameters = prepare_hyperparameters(config);
  if (!config.cuda_graph)
    return std::make_unique<EagerTrainer>(
        std::move(network), optimizer, metrics, indices, config.num_epochs,
        config.num_mini_batches, hyperparameters);
#ifdef __linux__
  return std::make_unique<CudaGraphTrainer>(
      std::move(network), optimizer, metrics, indices, config.num_epochs,
      config.num_mini_batches, hyperparameters, std::move(initial_batch));
#else
  (void)initial_batch;
  throw std::runtime_error(
      "cuda_graph is only supported on Linux; set cuda_graph=false.");
#endif
}

}  // namespace training
