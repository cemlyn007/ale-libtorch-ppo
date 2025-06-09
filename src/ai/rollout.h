#include "ale/ale_interface.hpp"
#include "buffer.h"
#include <filesystem>

namespace ai::rollout {

class Rollout {
public:
  Rollout(std::filesystem::path rom_path, size_t horizon, int num_episodes,
          int max_steps);
  void rollout();
  ale::Action select_action();
  void get_observation();

private:
  ale::ALEInterface ale_;
  std::string rom_path_;
  ai::buffer::Buffer buffer_;
  std::vector<unsigned char> observation_;
  int num_episodes_;
  int max_steps_;
  int current_episode_ = 0;
  int current_step_ = 0;
  bool is_terminal_ = false;
  bool is_truncated_ = false;
};

} // namespace ai::rollout