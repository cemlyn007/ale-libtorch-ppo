#include "ale/ale_interface.hpp"
#include <filesystem>

namespace ai::rollout {

class Rollout {
public:
  Rollout(std::filesystem::path rom_path, int num_episodes, int max_steps);
  void rollout();

private:
  ale::ALEInterface ale_;
  std::string rom_path_;
  int num_episodes_;
  int max_steps_;
  int current_episode_ = 0;
  int current_step_ = 0;
  bool is_done_ = false;
};

} // namespace ai::rollout