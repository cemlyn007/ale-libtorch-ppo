#include "ale/ale_interface.hpp"
#include "buffer.h"
#include <filesystem>
#include <functional>
#include <torch/torch.h>

namespace ai::rollout {

struct Batch {
  torch::Tensor observations;
  torch::Tensor actions;
  torch::Tensor rewards;
  torch::Tensor terminals;
  torch::Tensor truncations;
};

struct ActionResult {
  ale::Action action;
  torch::Tensor logits;
  torch::Tensor value;
};

class Rollout {
public:
  Rollout(std::filesystem::path rom_path, size_t horizon, size_t num_episodes,
          size_t max_steps, size_t frame_stack,
          std::function<ActionResult(const torch::Tensor &)> action_selector);
  Batch rollout();
  ActionResult select_action();
  void get_reset_observation();
  void get_observation();

private:
  ale::ALEInterface ale_;
  std::string rom_path_;
  ai::buffer::Buffer buffer_;
  std::vector<unsigned char> observation_;
  size_t horizon_;
  size_t num_episodes_;
  size_t frame_stack_;
  int max_steps_;
  int current_episode_ = 0;
  int current_step_ = 0;
  bool is_terminal_ = false;
  bool is_truncated_ = false;
  std::function<ActionResult(const torch::Tensor &)> action_selector_;
};

} // namespace ai::rollout