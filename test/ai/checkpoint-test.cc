#include "ai/checkpoint.h"

#include <torch/torch.h>

#include "gtest/gtest.h"

namespace {

// A tiny stand-in for the training network: ai::checkpoint::save/load take the
// torch::nn::Module base, so the round trip is independent of the real
// architecture and a couple of linear layers exercise every code path.
struct TinyNetImpl : torch::nn::Module {
  TinyNetImpl() : fc1(4, 3), fc2(3, 2) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }
  torch::nn::Linear fc1, fc2;
};
TORCH_MODULE(TinyNet);

// Pin every gradient to ones so each optimizer step is deterministic and the
// only thing distinguishing two runs is the restored optimizer state.
void step_with_unit_grads(TinyNet &net, torch::optim::Adam &optimizer) {
  for (auto &p : net->parameters()) {
    p.mutable_grad() = torch::ones_like(p);
  }
  optimizer.step();
}

}  // namespace

// Gives each test its own scratch directory under the (Bazel-sandboxed) test
// temp dir, created fresh and removed on teardown so checkpoints never leak or
// collide between tests.
class CheckpointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = std::filesystem::path(::testing::TempDir()) /
           ("checkpoint_test_" + std::string(info->name()));
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }
  std::filesystem::path path(const std::string &name) const {
    return dir_ / name;
  }

 private:
  std::filesystem::path dir_;
};

TEST_F(CheckpointTest, RoundTripsModelParameters) {
  torch::manual_seed(0);
  TinyNet saved;
  torch::optim::Adam optimizer(saved->parameters(),
                               torch::optim::AdamOptions(0.1));

  std::vector<torch::Tensor> expected;
  for (const auto &p : saved->parameters()) {
    expected.push_back(p.detach().clone());
  }

  const auto checkpoint_path = path("model_round_trip.pt");
  ai::checkpoint::save(checkpoint_path, *saved, optimizer,
                       {/*next_rollout_index=*/0,
                        /*best_return=*/0.0,
                        /*global_step=*/0});

  // A fresh network starts from different random weights; load must overwrite
  // them with the saved ones.
  TinyNet restored;
  torch::optim::Adam restored_optimizer(restored->parameters(),
                                        torch::optim::AdamOptions(0.1));
  ai::checkpoint::load(checkpoint_path, *restored, restored_optimizer,
                       torch::kCPU);

  const auto restored_params = restored->parameters();
  ASSERT_EQ(restored_params.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_TRUE(torch::equal(restored_params[i], expected[i]))
        << "Parameter " << i << " did not round trip";
  }
}

TEST_F(CheckpointTest, RoundTripsCheckpointScalars) {
  TinyNet net;
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(0.1));

  const ai::checkpoint::Checkpoint original{/*next_rollout_index=*/7,
                                            /*best_return=*/12.5,
                                            /*global_step=*/2048};
  const auto checkpoint_path = path("scalars_round_trip.pt");
  ai::checkpoint::save(checkpoint_path, *net, optimizer, original);

  TinyNet restored;
  torch::optim::Adam restored_optimizer(restored->parameters(),
                                        torch::optim::AdamOptions(0.1));
  const ai::checkpoint::Checkpoint loaded = ai::checkpoint::load(
      checkpoint_path, *restored, restored_optimizer, torch::kCPU);

  EXPECT_EQ(loaded.next_rollout_index, original.next_rollout_index);
  EXPECT_EQ(loaded.global_step, original.global_step);
  EXPECT_DOUBLE_EQ(loaded.best_return, original.best_return);
}

TEST_F(CheckpointTest, RestoresOptimizerStateSoTrainingContinues) {
  torch::manual_seed(0);
  TinyNet net;
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(0.1).eps(1e-5));

  // Warm up so Adam accumulates first/second moments and a non-zero step count.
  step_with_unit_grads(net, optimizer);
  step_with_unit_grads(net, optimizer);

  const auto checkpoint_path = path("optimizer_state.pt");
  ai::checkpoint::save(checkpoint_path, *net, optimizer,
                       {/*next_rollout_index=*/2, /*best_return=*/0.0,
                        /*global_step=*/0});

  // One more step on the original run is the ground truth a resumed run must
  // reproduce exactly.
  step_with_unit_grads(net, optimizer);
  std::vector<torch::Tensor> expected;
  for (const auto &p : net->parameters()) {
    expected.push_back(p.detach().clone());
  }

  TinyNet resumed;
  torch::optim::Adam resumed_optimizer(
      resumed->parameters(), torch::optim::AdamOptions(0.1).eps(1e-5));
  ai::checkpoint::load(checkpoint_path, *resumed, resumed_optimizer,
                       torch::kCPU);
  step_with_unit_grads(resumed, resumed_optimizer);

  const auto resumed_params = resumed->parameters();
  ASSERT_EQ(resumed_params.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Without restored Adam moments and step count, bias correction on this
    // step would diverge from the original run.
    EXPECT_TRUE(torch::allclose(resumed_params[i], expected[i], 1e-6, 1e-8))
        << "Resumed step diverged for parameter " << i;
  }
}

TEST_F(CheckpointTest, CheckpointerWritesBestOnlyOnImprovement) {
  TinyNet net;
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(0.1));
  std::vector<std::string> announced;
  ai::checkpoint::Checkpointer checkpointer(
      path(""), /*interval=*/100,
      [&](size_t, const std::string &text) { announced.push_back(text); });

  checkpointer.on_rollout_end(0, 10, 5.0, *net, optimizer);
  ASSERT_TRUE(std::filesystem::exists(path("best.pt")));
  EXPECT_DOUBLE_EQ(checkpointer.best_return(), 5.0);

  // A worse rollout and an episode-free rollout must both leave best.pt alone.
  checkpointer.on_rollout_end(1, 20, 4.0, *net, optimizer);
  checkpointer.on_rollout_end(2, 30, std::nullopt, *net, optimizer);
  EXPECT_DOUBLE_EQ(checkpointer.best_return(), 5.0);
  ASSERT_EQ(announced.size(), 1u);
  EXPECT_NE(announced[0].find("best.pt"), std::string::npos);

  checkpointer.on_rollout_end(3, 40, 6.0, *net, optimizer);
  EXPECT_DOUBLE_EQ(checkpointer.best_return(), 6.0);
  EXPECT_EQ(announced.size(), 2u);
  EXPECT_FALSE(std::filesystem::exists(path("latest.pt")));
}

TEST_F(CheckpointTest, CheckpointerWritesLatestOnInterval) {
  TinyNet net;
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(0.1));
  ai::checkpoint::Checkpointer checkpointer(path(""), /*interval=*/2);

  checkpointer.on_rollout_end(0, 10, std::nullopt, *net, optimizer);
  EXPECT_FALSE(std::filesystem::exists(path("latest.pt")));

  // Interval counts completed rollouts, so the second rollout (index 1) hits.
  checkpointer.on_rollout_end(1, 20, std::nullopt, *net, optimizer);
  ASSERT_TRUE(std::filesystem::exists(path("latest.pt")));

  TinyNet restored;
  torch::optim::Adam restored_optimizer(restored->parameters(),
                                        torch::optim::AdamOptions(0.1));
  const ai::checkpoint::Checkpoint loaded = ai::checkpoint::load(
      path("latest.pt"), *restored, restored_optimizer, torch::kCPU);
  EXPECT_EQ(loaded.next_rollout_index, 2u);
  EXPECT_EQ(loaded.global_step, 20u);
}

TEST_F(CheckpointTest, CheckpointerDisabledWhenIntervalZero) {
  TinyNet net;
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(0.1));
  std::vector<std::string> announced;
  ai::checkpoint::Checkpointer checkpointer(
      path(""), /*interval=*/0,
      [&](size_t, const std::string &text) { announced.push_back(text); });

  checkpointer.on_rollout_end(0, 10, 5.0, *net, optimizer);
  EXPECT_FALSE(std::filesystem::exists(path("best.pt")));
  EXPECT_FALSE(std::filesystem::exists(path("latest.pt")));
  EXPECT_TRUE(announced.empty());
}
