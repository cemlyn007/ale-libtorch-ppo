#pragma once
#include <torch/torch.h>

#include <memory>

#include "ai/buffer.h"
#include "ai/ppo/train.h"
#include "ai/ppo/trainer.h"
#include "training/config.h"
#include "training/network.h"

namespace training {

// buffer::Batch -> ppo::train::Batch: flatten the [time, env] rollout into one
// batch dimension and convert the stored behaviour logits to log-probabilities.
// Exposed because the CUDA-graph path needs an initial batch to capture with.
ai::ppo::train::Batch prepare_batch(ai::buffer::Batch &batch);

// Builds the training strategy once, up front: the CUDA-graph path captures its
// graph here (so unsupported platforms are rejected eagerly rather than failing
// inside the loop). initial_batch is the persistent capture input; the eager
// path ignores it.
std::unique_ptr<ai::ppo::Trainer> make_trainer(
    const Config &config, Network network, torch::optim::Optimizer &optimizer,
    ai::ppo::train::Metrics &metrics, torch::Tensor &indices,
    ai::ppo::train::Batch initial_batch);

}  // namespace training
