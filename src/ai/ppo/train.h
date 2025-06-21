#include "ai/ppo/losses.h"
#include <torch/torch.h>
namespace ai::ppo::train {

template <typename T>
concept NetworkModel = requires(T &model, torch::Tensor input) {
  // Requires train() method
  { model->train() };

  // Requires forward() method returning object with logits and value members
  { model->forward(input) };
  { model->forward(input).logits } -> std::convertible_to<torch::Tensor>;
  { model->forward(input).value } -> std::convertible_to<torch::Tensor>;

  // Requires parameters method compatible with
  // torch::nn::utils::clip_grad_norm_
  { model->parameters() } -> std::same_as<std::vector<torch::Tensor>>;
};

struct Hyperparameters {
  float clip_param;
  float value_loss_coef;
  float entropy_coef;
  float max_gradient_norm;
};

struct Batch {
  torch::Tensor observations;
  torch::Tensor actions;
  torch::Tensor logits;
  torch::Tensor advantages;
  torch::Tensor returns;
  torch::Tensor masks;

  Batch slice(int64_t start, int64_t end) const {
    return {observations.slice(0, start, end), actions.slice(0, start, end),
            logits.slice(0, start, end),       advantages.slice(0, start, end),
            returns.slice(0, start, end),      masks.slice(0, start, end)};
  }
};

struct Metrics {
  torch::Tensor loss;
  torch::Tensor clipped_losses;
  torch::Tensor value_losses;
  torch::Tensor entropies;
  torch::Tensor ratio;
  torch::Tensor total_losses;
  torch::Tensor advantages;
  torch::Tensor returns;
  torch::Tensor masks;
  torch::Tensor clipped_gradients;

  Metrics(int64_t num_epochs, int64_t num_mini_batches, int64_t mini_batch_size,
          const torch::Device &device) {
    auto options = torch::TensorOptions().device(device);
    loss =
        torch::empty({num_epochs, num_mini_batches, mini_batch_size}, options);
    clipped_losses = torch::empty_like(loss);
    value_losses = torch::empty_like(loss);
    entropies = torch::empty_like(loss);
    ratio = torch::empty_like(loss);
    total_losses = torch::empty_like(loss);
    advantages = torch::empty_like(loss);
    returns = torch::empty_like(loss);
    masks = torch::empty({num_epochs, num_mini_batches, mini_batch_size},
                         options.dtype(torch::kBool));
    clipped_gradients = torch::empty({num_epochs, num_mini_batches});
  }
};

template <NetworkModel Network>
ai::ppo::losses::Metrics
compute_loss(Network &network, const torch::Tensor &observations,
             const torch::Tensor &actions, const torch::Tensor &advantages,
             const torch::Tensor &old_logits, const torch::Tensor &returns,
             const torch::Tensor &masks, float clip_param,
             float value_loss_coef, float entropy_coef) {
  auto output = network->forward(observations);
  auto logits = output.logits;
  auto values = output.value;
  return ai::ppo::losses::compute(logits, old_logits, actions, advantages,
                                  values, returns, masks, clip_param,
                                  value_loss_coef, entropy_coef);
}

struct MiniBatchUpdateResult {
  ai::ppo::losses::Metrics ppo;
  double clipped_gradient;
};

template <NetworkModel Network>
MiniBatchUpdateResult
mini_batch_update(Network &network, torch::optim::Optimizer &optimizer,
                  const Batch &batch, Hyperparameters &hyperparameters) {
  {
    auto ppo_metrics = compute_loss(
        network, batch.observations, batch.actions, batch.advantages,
        batch.logits, batch.returns, batch.masks, hyperparameters.clip_param,
        hyperparameters.value_loss_coef, hyperparameters.entropy_coef);
    optimizer.zero_grad();
    ppo_metrics.loss.backward();
    auto clipped_gradient = torch::nn::utils::clip_grad_norm_(
        network->parameters(), hyperparameters.max_gradient_norm, 2.0, true);
    optimizer.step();

    return {ppo_metrics, clipped_gradient};
  }
}

template <NetworkModel Network>
void train(Network &network, torch::optim::Optimizer &optimizer,
           Metrics &metrics, torch::Tensor &indices, Batch &batch,
           size_t num_epochs, size_t num_mini_batches,
           Hyperparameters &hyperparameters) {
  network->train();
  size_t mini_batch_size =
      indices.size(0) / num_mini_batches; // Assuming batch size is divisible
  for (size_t epoch_index = 0; epoch_index < num_epochs; epoch_index++) {
    torch::randperm_out(indices, batch.observations.size(0));
    for (size_t mini_batch_index = 0; mini_batch_index < num_mini_batches;
         mini_batch_index++) {
      auto start = mini_batch_index * mini_batch_size;
      auto end = start + mini_batch_size;
      Batch mini_batch = batch.slice(start, end);
      MiniBatchUpdateResult result =
          mini_batch_update(network, optimizer, mini_batch, hyperparameters);
      const torch::indexing::TensorIndex indices(
          {{epoch_index, mini_batch_index}});
      metrics.loss.index_put_(indices, result.ppo.loss.reshape({1}));
      metrics.clipped_losses.index_put_(indices, result.ppo.clipped_losses);
      metrics.value_losses.index_put_(indices, result.ppo.value_losses);
      metrics.entropies.index_put_(indices, result.ppo.entropies);
      metrics.ratio.index_put_(indices, result.ppo.ratio);
      metrics.total_losses.index_put_(indices, result.ppo.total_losses);
      metrics.clipped_gradients.index_put_(indices, result.clipped_gradient);
      metrics.advantages.index_put_(indices, mini_batch.advantages);
      metrics.returns.index_put_(indices, mini_batch.returns);
      metrics.masks.index_put_(indices, mini_batch.masks);
    }
  }
}
} // namespace ai::ppo::train