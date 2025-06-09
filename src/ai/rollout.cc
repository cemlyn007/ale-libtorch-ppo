#include "rollout.h"
namespace ai::rollout {

Rollout::Rollout(std::filesystem::path rom_path, int num_episodes,
                 int max_steps)
    : rom_path_(rom_path), num_episodes_(num_episodes), max_steps_(max_steps) {
  ale_.loadROM(rom_path_);
}

void Rollout::rollout() {
  while (current_episode_ < num_episodes_) {
    ale_.reset_game();
    current_step_ = 0;
    is_done_ = false;
    while (current_step_ < max_steps_ && !is_done_) {
      auto action =
          ale_.getMinimalActionSet()[rand() %
                                     ale_.getMinimalActionSet().size()];
      ale_.act(action);
      is_done_ = ale_.game_over();
      current_step_++;
    }
    current_episode_++;
  }
}
} // namespace ai::rollout