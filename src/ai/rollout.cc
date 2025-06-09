#include "rollout.h"

namespace ai::rollout {

Rollout::Rollout(std::filesystem::path rom_path, size_t horizon,
                 int num_episodes, int max_steps)
    : ale_(), rom_path_(rom_path), buffer_([&] {
        ale::ALEInterface tmp_ale;
        tmp_ale.loadROM(rom_path);
        auto screen = tmp_ale.getScreen();
        return ai::buffer::Buffer(horizon, {screen.width(), screen.height()},
                                  1);
      }()),
      num_episodes_(num_episodes), max_steps_(max_steps) {
  ale_.loadROM(rom_path_);
}

ale::Action Rollout::select_action() {
  return ale_.getMinimalActionSet()[rand() % ale_.getMinimalActionSet().size()];
}

void Rollout::get_observation() { ale_.getScreenGrayscale(observation_); }

void Rollout::rollout() {
  while (current_episode_ < num_episodes_) {
    ale_.reset_game();

    get_observation();

    current_step_ = 0;
    is_done_ = false;
    while (current_step_ < max_steps_ && !is_done_) {
      auto action = select_action();
      auto reward = ale_.act(action);

      buffer_.add(observation_, action, reward);

      get_observation();
      std::cout << "Episode: " << current_episode_
                << ", Step: " << current_step_ << ", Action: " << action
                << ", Reward: " << reward << std::endl;
      is_done_ = ale_.game_over();
      current_step_++;
    }
    current_episode_++;
  }
}
} // namespace ai::rollout