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
};

// TODO
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

template <NetworkModel Network>
void mini_batch_update(torch::Device &device, Network &network,
                       torch::optim::Optimizer &optimizer, Metrics &metrics,
                       const torch::Tensor &indices, const Batch &batch,
                       size_t mini_batch_size, long j, long k,
                       Hyperparameters &hyperparameters) {
  {
    auto start = k * mini_batch_size;
    auto end = start + mini_batch_size;
    const auto &mini_indices = indices.slice(0, start, end);

    torch::Tensor mini_observations =
        batch.observations.index_select(0, mini_indices);
    torch::Tensor mini_actions = batch.actions.index_select(0, mini_indices);
    torch::Tensor mini_advantages =
        batch.advantages.index_select(0, mini_indices);
    torch::Tensor mini_logits = batch.logits.index_select(0, mini_indices);
    torch::Tensor mini_returns = batch.returns.index_select(0, mini_indices);
    torch::Tensor mini_masks = batch.masks.index_select(0, mini_indices);

    auto ppo_metrics = compute_loss(
        network, mini_observations, mini_actions, mini_advantages, mini_logits,
        mini_returns, mini_masks, hyperparameters.clip_param,
        hyperparameters.value_loss_coef, hyperparameters.entropy_coef);
    optimizer.zero_grad();
    ppo_metrics.loss.backward();
    auto clipped_gradient = torch::nn::utils::clip_grad_norm_(
        network->parameters(), hyperparameters.max_gradient_norm, 2.0, true);
    optimizer.step();

    const torch::indexing::TensorIndex indices({{j, k}});
    metrics.loss.index_put_(indices, ppo_metrics.loss.reshape({1}));
    metrics.clipped_losses.index_put_(indices, ppo_metrics.clipped_losses);
    metrics.value_losses.index_put_(indices, ppo_metrics.value_losses);
    metrics.entropies.index_put_(indices, ppo_metrics.entropies);
    metrics.ratio.index_put_(indices, ppo_metrics.ratio);
    metrics.total_losses.index_put_(indices, ppo_metrics.total_losses);
    metrics.advantages.index_put_(indices, mini_advantages);
    metrics.returns.index_put_(indices, mini_returns);
    metrics.masks.index_put_(indices, mini_masks);
    metrics.clipped_gradients.index_put_(indices, clipped_gradient);
  }
}

template <NetworkModel Network>
void train(torch::Device &device, Network &network,
           torch::optim::Optimizer &optimizer, Metrics &metrics,
           torch::Tensor &indices, Batch &batch, size_t num_epochs,
           size_t num_mini_batches, Hyperparameters &hyperparameters) {
  network->train();
  size_t mini_batch_size = indices.size(0) / num_mini_batches;
  for (size_t j = 0; j < num_epochs; j++) {
    torch::randperm_out(indices, batch.observations.size(0));
    for (size_t k = 0; k < num_mini_batches; k++)
      mini_batch_update(device, network, optimizer, metrics, indices, batch,
                        mini_batch_size, static_cast<long>(j),
                        static_cast<long>(k), hyperparameters);
  }
}
} // namespace ai::ppo::train