#include <gtest/gtest.h>
#include <torch/torch.h>

#include "ai/ppo/train.h"

namespace {

struct PolicyOutput {
  torch::Tensor logits;
  torch::Tensor value;
};

// Minimal model satisfying ai::ppo::train::NetworkModel.
struct PolicyImpl : torch::nn::Module {
  PolicyImpl(int64_t observation_size, int64_t action_size)
      : linear(register_module(
            "linear", torch::nn::Linear(observation_size, action_size + 1))) {}

  PolicyOutput forward(const torch::Tensor &observations) {
    auto output = linear(observations);
    return {output.slice(-1, 0, output.size(-1) - 1),
            output.select(-1, output.size(-1) - 1)};
  }

  torch::nn::Linear linear;
};
TORCH_MODULE(Policy);

TEST(PpoTrainTest, MiniBatchesAreShuffledPermutationsEachEpoch) {
  torch::manual_seed(0);
  constexpr int64_t size = 32;
  constexpr int64_t observation_size = 4;
  constexpr int64_t action_size = 3;
  constexpr size_t num_epochs = 3;
  constexpr size_t num_mini_batches = 4;
  constexpr int64_t mini_batch_size = size / num_mini_batches;

  Policy network(observation_size, action_size);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(1e-3));

  // Advantages tag each sample with its row index so metrics reveal which
  // rows each mini-batch saw.
  ai::ppo::train::Batch batch = {
      torch::randn({size, observation_size}),
      torch::randint(0, action_size, {size}, torch::kLong),
      ai::ppo::losses::normalize_logits(torch::randn({size, action_size})),
      torch::arange(size, torch::kFloat),
      torch::randn({size}),
      torch::ones({size}, torch::kBool)};

  torch::Tensor indices = torch::empty({size}, torch::kLong);
  ai::ppo::train::Metrics metrics(num_epochs, num_mini_batches, mini_batch_size,
                                  torch::Device(torch::kCPU));
  ai::ppo::train::Hyperparameters hyperparameters = {
      0.2f, 0.5f, 0.01f, 0.5f,
      /*shuffle_mini_batches=*/true};

  ai::ppo::train::train(network, optimizer, metrics, indices, batch, num_epochs,
                        num_mini_batches, hyperparameters);

  auto identity = torch::arange(size, torch::kFloat);
  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
    auto seen = metrics.advantages[epoch].flatten();
    // Every sample is visited exactly once per epoch...
    EXPECT_TRUE(torch::equal(std::get<0>(torch::sort(seen)), identity))
        << "epoch " << epoch << " is not a permutation of the batch";
    // ...and not in the contiguous env-major order.
    EXPECT_FALSE(torch::equal(seen, identity))
        << "epoch " << epoch << " mini-batches were not shuffled";
  }
  EXPECT_FALSE(torch::equal(metrics.advantages[0], metrics.advantages[1]))
      << "epochs reused the same permutation";
}

TEST(PpoTrainTest, ContiguousMiniBatchesWhenShufflingDisabled) {
  torch::manual_seed(0);
  constexpr int64_t size = 32;
  constexpr int64_t observation_size = 4;
  constexpr int64_t action_size = 3;
  constexpr size_t num_epochs = 2;
  constexpr size_t num_mini_batches = 4;
  constexpr int64_t mini_batch_size = size / num_mini_batches;

  Policy network(observation_size, action_size);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(1e-3));

  ai::ppo::train::Batch batch = {
      torch::randn({size, observation_size}),
      torch::randint(0, action_size, {size}, torch::kLong),
      ai::ppo::losses::normalize_logits(torch::randn({size, action_size})),
      torch::arange(size, torch::kFloat),
      torch::randn({size}),
      torch::ones({size}, torch::kBool)};

  torch::Tensor indices = torch::empty({size}, torch::kLong);
  ai::ppo::train::Metrics metrics(num_epochs, num_mini_batches, mini_batch_size,
                                  torch::Device(torch::kCPU));
  ai::ppo::train::Hyperparameters hyperparameters = {
      0.2f, 0.5f, 0.01f, 0.5f, /*shuffle_mini_batches=*/false};

  ai::ppo::train::train(network, optimizer, metrics, indices, batch, num_epochs,
                        num_mini_batches, hyperparameters);

  auto identity = torch::arange(size, torch::kFloat);
  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
    EXPECT_TRUE(torch::equal(metrics.advantages[epoch].flatten(), identity))
        << "epoch " << epoch
        << " did not visit samples in contiguous env-major order";
  }
}

#ifdef __linux__
TEST(PpoTrainTest, CudaGraphCaptureShufflesMiniBatches) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  torch::Device device(torch::kCUDA);
  torch::manual_seed(0);
  constexpr int64_t size = 32;
  constexpr int64_t observation_size = 4;
  constexpr int64_t action_size = 3;
  constexpr size_t num_epochs = 2;
  constexpr size_t num_mini_batches = 4;
  constexpr int64_t mini_batch_size = size / num_mini_batches;

  Policy network(observation_size, action_size);
  network->to(device);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(1e-3));

  auto options = torch::TensorOptions().device(device);
  ai::ppo::train::Batch batch = {
      torch::randn({size, observation_size}, options),
      torch::randint(0, action_size, {size}, options.dtype(torch::kLong)),
      ai::ppo::losses::normalize_logits(
          torch::randn({size, action_size}, options)),
      torch::arange(size, options.dtype(torch::kFloat)),
      torch::randn({size}, options),
      torch::ones({size}, options.dtype(torch::kBool))};

  torch::Tensor indices = torch::empty({size}, options.dtype(torch::kLong));
  ai::ppo::train::Metrics metrics(num_epochs, num_mini_batches, mini_batch_size,
                                  device);
  ai::ppo::train::Hyperparameters hyperparameters = {
      0.2f, 0.5f, 0.01f, 0.5f,
      /*shuffle_mini_batches=*/true};

  at::cuda::CUDAGraph graph;
  network->train();
  ai::ppo::train::capture_train_cuda_graph(graph, network, optimizer, metrics,
                                           indices, batch, num_epochs,
                                           num_mini_batches, hyperparameters,
                                           /*num_warmup_iters=*/3);

  auto check_permutations = [&](const torch::Tensor &advantages) {
    auto identity = torch::arange(size, torch::kFloat);
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
      auto seen = advantages[epoch].flatten();
      EXPECT_TRUE(torch::equal(std::get<0>(torch::sort(seen)), identity))
          << "epoch " << epoch << " is not a permutation of the batch";
      EXPECT_FALSE(torch::equal(seen, identity))
          << "epoch " << epoch << " mini-batches were not shuffled";
    }
  };

  ai::ppo::train::train_cuda_graph(graph);
  torch::cuda::synchronize();
  auto first_replay = metrics.advantages.to(torch::kCPU);
  check_permutations(first_replay);

  ai::ppo::train::train_cuda_graph(graph);
  torch::cuda::synchronize();
  auto second_replay = metrics.advantages.to(torch::kCPU);
  check_permutations(second_replay);
  EXPECT_FALSE(torch::equal(first_replay, second_replay))
      << "graph replays reused the same permutations";
}
#endif

TEST(PpoTrainTest, RejectsIndivisibleMiniBatchCount) {
  Policy network(4, 3);
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(1e-3));
  ai::ppo::train::Batch batch = {
      torch::randn({10, 4}),
      torch::randint(0, 3, {10}, torch::kLong),
      ai::ppo::losses::normalize_logits(torch::randn({10, 3})),
      torch::randn({10}),
      torch::randn({10}),
      torch::ones({10}, torch::kBool)};
  torch::Tensor indices = torch::empty({10}, torch::kLong);
  ai::ppo::train::Metrics metrics(1, 3, 3, torch::Device(torch::kCPU));
  ai::ppo::train::Hyperparameters hyperparameters = {
      0.2f, 0.5f, 0.01f, 0.5f,
      /*shuffle_mini_batches=*/true};
  EXPECT_THROW(ai::ppo::train::train(network, optimizer, metrics, indices,
                                     batch, 1, 3, hyperparameters),
               std::runtime_error);
}

}  // namespace
