#include "ai/environment/episode_life.h"

namespace ai::environment {

EpisodeLife::EpisodeLife(std::unique_ptr<VirtualEnvironment> env)
    : env_(std::move(env)), lives_(0) {}

ScreenBuffer EpisodeLife::reset() {
  ScreenBuffer observation;
  if (lives_ > 0 && !env_->get_interface().game_over(false) &&
      !env_->get_interface().game_truncated()) {
    auto step_result = env_->step(ale::Action::PLAYER_A_NOOP);
    observation = step_result.observation;
  } else {
    observation = env_->reset();
  }
  lives_ = env_->get_interface().lives();
  if (lives_ <= 0)
    throw std::runtime_error("No lives left in the environment.");
  auto fire_result = env_->step(ale::Action::PLAYER_A_FIRE);
  observation = fire_result.observation;
  return observation;
}

Step EpisodeLife::step(const ale::Action &action) {
  auto result = env_->step(action);
  int current_lives = env_->get_interface().lives();
  bool life_lost = current_lives < lives_;
  lives_ = current_lives;
  result.truncated |= (!result.terminated && life_lost);
  return result;
}

ale::ALEInterface &EpisodeLife::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
