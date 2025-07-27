#include "ai/gae.h"
#include "gtest/gtest.h"
#include <torch/torch.h>
#include <vector>

TEST(GAETest, SimpleNoTerminalsNoTruncations) {
  // 1 env, 3 steps, no terminals, no truncations, no episode_starts
  torch::Tensor rewards = torch::tensor({{1.0, 1.0, 1.0}});
  torch::Tensor values = torch::tensor({{0.5, 0.5, 0.5}});
  torch::Tensor next_values = torch::tensor({0.5});
  torch::Tensor terminals = torch::zeros({1, 3}, torch::kBool);
  torch::Tensor truncations = torch::zeros({1, 3}, torch::kBool);
  torch::Tensor episode_starts = torch::zeros({1, 3}, torch::kBool);
  float gamma = 0.99;
  float lambda = 0.95;
  torch::Tensor advantages = torch::zeros({1, 3});
  ai::gae::gae(advantages, rewards, values, next_values, terminals, truncations,
               episode_starts, gamma, lambda);

  // Compute expected advantages inline
  std::vector<float> exp_vec(3);
  float last = 0.0f;
  float nv = next_values[0].item<float>();
  for (int i = 2; i >= 0; --i) {
    float r = rewards.index({0, i}).item<float>();
    float v = values.index({0, i}).item<float>();
    float delta = r + gamma * nv - v;
    float adv = delta + gamma * lambda * last;
    exp_vec[i] = adv;
    last = adv;
    nv = v;
  }
  // Compare each advantage element-wise for precise failure reporting
  for (int idx = 0; idx < 3; ++idx) {
    float act = advantages.index({0, idx}).item<float>();
    float exp = exp_vec[idx];
    EXPECT_NEAR(act, exp, 1e-5) << "Mismatch at index " << idx;
  }
}

TEST(GAETest, TerminalHandling) {
  // 1 env, 3 steps, terminal at last step
  torch::Tensor rewards = torch::tensor({{1.0, 1.0, 1.0}});
  torch::Tensor values = torch::tensor({{0.5, 0.5, 0.5}});
  torch::Tensor next_values = torch::tensor({0.0});
  torch::Tensor terminals = torch::tensor({{0, 0, 1}}, torch::kBool);
  torch::Tensor truncations = torch::zeros({1, 3}, torch::kBool);
  torch::Tensor episode_starts = torch::zeros({1, 3}, torch::kBool);
  float gamma = 0.99;
  float lambda = 0.95;
  torch::Tensor advantages = torch::zeros({1, 3});
  ai::gae::gae(advantages, rewards, values, next_values, terminals, truncations,
               episode_starts, gamma, lambda);

  std::vector<float> exp_vec(3);
  float last = 0.0f;
  float nv = next_values[0].item<float>();
  for (int i = 2; i >= 0; --i) {
    float r = rewards.index({0, i}).item<float>();
    float v = values.index({0, i}).item<float>();
    bool term = terminals.index({0, i}).item<bool>();
    float adv;
    if (term) {
      adv = r - v;
    } else {
      float delta = r + gamma * nv - v;
      adv = delta + gamma * lambda * last;
    }
    exp_vec[i] = adv;
    last = adv;
    nv = v;
  }
  for (int idx = 0; idx < 3; ++idx) {
    float act = advantages.index({0, idx}).item<float>();
    float exp = exp_vec[idx];
    EXPECT_NEAR(act, exp, 1e-5) << "Mismatch at index " << idx;
  }
}

TEST(GAETest, TruncationHandling) {
  // 1 env, 3 steps, truncation at last step
  torch::Tensor rewards = torch::tensor({{1.0, 1.0, 1.0}});
  torch::Tensor values = torch::tensor({{0.5, 0.5, 0.5}});
  torch::Tensor next_values = torch::tensor({0.0});
  torch::Tensor terminals = torch::zeros({1, 3}, torch::kBool);
  torch::Tensor truncations = torch::tensor({{0, 0, 1}}, torch::kBool);
  torch::Tensor episode_starts = torch::zeros({1, 3}, torch::kBool);
  float gamma = 0.99;
  float lambda = 0.95;
  torch::Tensor advantages = torch::zeros({1, 3});
  ai::gae::gae(advantages, rewards, values, next_values, terminals, truncations,
               episode_starts, gamma, lambda);

  std::vector<float> exp_vec(3);
  float last = 0.0f;
  float nv = next_values[0].item<float>();
  for (int i = 2; i >= 0; --i) {
    float r = rewards.index({0, i}).item<float>();
    float v = values.index({0, i}).item<float>();
    bool trunc = truncations.index({0, i}).item<bool>();
    float delta = r + gamma * nv - v;
    float adv = trunc ? delta : (delta + gamma * lambda * last);
    exp_vec[i] = adv;
    last = adv;
    nv = v;
  }
  for (int idx = 0; idx < 3; ++idx) {
    float act = advantages.index({0, idx}).item<float>();
    float exp = exp_vec[idx];
    EXPECT_NEAR(act, exp, 1e-5) << "Mismatch at index " << idx;
  }
}

TEST(GAETest, EpisodeStartHandling) {
  // 1 env, 3 steps, episode_start at step 1
  torch::Tensor rewards = torch::tensor({{1.0, 1.0, 1.0}});
  torch::Tensor values = torch::tensor({{0.5, 0.5, 0.5}});
  torch::Tensor next_values = torch::tensor({0.5});
  torch::Tensor terminals = torch::zeros({1, 3}, torch::kBool);
  torch::Tensor truncations = torch::zeros({1, 3}, torch::kBool);
  torch::Tensor episode_starts = torch::tensor({{0, 1, 0}}, torch::kBool);
  float gamma = 0.99;
  float lambda = 0.95;
  torch::Tensor advantages = torch::zeros({1, 3});
  ai::gae::gae(advantages, rewards, values, next_values, terminals, truncations,
               episode_starts, gamma, lambda);

  std::vector<float> exp_vec(3);
  float last = 0.0f;
  float nv = next_values[0].item<float>();
  for (int i = 2; i >= 0; --i) {
    float r = rewards.index({0, i}).item<float>();
    float v = values.index({0, i}).item<float>();
    bool start = episode_starts.index({0, i}).item<bool>();
    float delta = r + gamma * nv - v;
    float adv = start ? 0.0f : (delta + gamma * lambda * last);
    exp_vec[i] = adv;
    last = adv;
    nv = v;
  }
  for (int idx = 0; idx < 3; ++idx) {
    float act = advantages.index({0, idx}).item<float>();
    float exp = exp_vec[idx];
    EXPECT_NEAR(act, exp, 1e-5) << "Mismatch at index " << idx;
  }
}

TEST(GAETest, MultipleEnvironments) {
  // 2 envs, 3 steps, different patterns in each env
  torch::Tensor rewards = torch::tensor({{1.0, 1.0, 1.0}, {0.5, 0.5, 0.5}});
  torch::Tensor values = torch::tensor({{0.5, 0.5, 0.5}, {0.3, 0.3, 0.3}});
  torch::Tensor next_values = torch::tensor({0.5, 0.3});
  torch::Tensor terminals = torch::zeros({2, 3}, torch::kBool);
  terminals.index_put_({1, 2}, true); // Terminal at last step for env 2
  torch::Tensor truncations = torch::zeros({2, 3}, torch::kBool);
  torch::Tensor episode_starts = torch::zeros({2, 3}, torch::kBool);
  episode_starts.index_put_({0, 1}, true); // Episode start at step 1 for env 1
  float gamma = 0.99;
  float lambda = 0.95;
  torch::Tensor advantages = torch::zeros({2, 3});

  // Run GAE
  ai::gae::gae(advantages, rewards, values, next_values, terminals, truncations,
               episode_starts, gamma, lambda);

  // Compute expected for each environment
  std::vector<std::vector<float>> exp(2, std::vector<float>(3));
  for (int j = 0; j < 2; ++j) {
    float last = 0.0f;
    float nv = next_values[j].item<float>();
    for (int i = 2; i >= 0; --i) {
      float r = rewards.index({j, i}).item<float>();
      float v = values.index({j, i}).item<float>();
      bool term = terminals.index({j, i}).item<bool>();
      bool trunc = truncations.index({j, i}).item<bool>();
      bool start = episode_starts.index({j, i}).item<bool>();
      float delta = r + gamma * nv - v;
      float adv;
      if (start) {
        adv = 0.0f;
      } else if (term) {
        adv = r - v;
      } else if (trunc) {
        adv = delta;
      } else {
        adv = delta + gamma * lambda * last;
      }
      exp[j][i] = adv;
      last = adv;
      nv = v;
    }
  }

  // Compare results
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 3; ++i) {
      float act = advantages.index({j, i}).item<float>();
      float ex = exp[j][i];
      EXPECT_NEAR(act, ex, 1e-5) << "Env " << j << ", step " << i
                                 << ": expected " << ex << " but got " << act;
    }
  }
}

TEST(GAETest, MixedStateFlags) {
  // Test sequence with terminal, truncation and episode start at different
  // positions
  torch::Tensor rewards = torch::tensor({{1.0, 1.0, 1.0, 1.0, 1.0}});
  torch::Tensor values = torch::tensor({{0.5, 0.5, 0.5, 0.5, 0.5}});
  torch::Tensor next_values = torch::tensor({0.5});
  torch::Tensor terminals = torch::zeros({1, 5}, torch::kBool);
  terminals.index_put_({0, 2}, true); // Terminal at middle step
  torch::Tensor truncations = torch::zeros({1, 5}, torch::kBool);
  truncations.index_put_({0, 4}, true); // Truncation at end
  torch::Tensor episode_starts = torch::zeros({1, 5}, torch::kBool);
  episode_starts.index_put_({0, 3}, true); // New episode after terminal
  float gamma = 0.99;
  float lambda = 0.95;
  torch::Tensor advantages = torch::zeros({1, 5});

  // Run GAE
  ai::gae::gae(advantages, rewards, values, next_values, terminals, truncations,
               episode_starts, gamma, lambda);

  // Compute expected advantages
  std::vector<float> exp_vec(5);
  float last = 0.0f;
  float nv = next_values[0].item<float>();
  for (int i = 4; i >= 0; --i) {
    float r = rewards.index({0, i}).item<float>();
    float v = values.index({0, i}).item<float>();
    bool term = terminals.index({0, i}).item<bool>();
    bool trunc = truncations.index({0, i}).item<bool>();
    bool start = episode_starts.index({0, i}).item<bool>();
    float delta = r + gamma * nv - v;
    float adv;
    if (start) {
      adv = 0.0f;
    } else if (term) {
      adv = r - v;
    } else if (trunc) {
      adv = delta;
    } else {
      adv = delta + gamma * lambda * last;
    }
    exp_vec[i] = adv;
    last = adv;
    nv = v;
  }

  // Compare results
  for (int i = 0; i < 5; ++i) {
    float act = advantages.index({0, i}).item<float>();
    EXPECT_NEAR(act, exp_vec[i], 1e-5)
        << "Step " << i << ": expected " << exp_vec[i] << " but got " << act;
  }
}