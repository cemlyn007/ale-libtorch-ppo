#include "ai/environment/episode_life.h"

namespace ai::environment {

EpisodeLife::EpisodeLife(std::unique_ptr<VirtualEnvironment> env)
    : env_(std::move(env)), lives_(0), game_over_(true) {}

ScreenBuffer EpisodeLife::reset() {
  ScreenBuffer observation;
  if (game_over_) {
    observation = env_->reset();
    game_over_ = false;
  } else {
    auto step_result = env_->step(ale::Action::PLAYER_A_NOOP);
    observation = step_result.observation;
    game_over_ = step_result.game_over;
    if (step_result.terminated || step_result.truncated) {
      observation = env_->reset();
      game_over_ = false;
    }
  }
  lives_ = env_->get_interface().lives();
  return observation;
}

Step EpisodeLife::step(const ale::Action &action) {
  if (game_over_)
    throw std::runtime_error("Cannot step in a game that is over.");
  if (lives_ <= 0)
    throw std::runtime_error("No lives left in the environment.");
  auto result = env_->step(action);
  int new_lives = env_->get_interface().lives();
  bool life_lost = new_lives < lives_;
  result.terminated |= life_lost;
  if (result.terminated)
    result.truncated = false;
  lives_ = new_lives;
  game_over_ = result.game_over;
  return result;
}

ale::ALEInterface &EpisodeLife::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
