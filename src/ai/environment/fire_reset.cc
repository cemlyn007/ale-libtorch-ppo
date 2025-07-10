#include "ai/environment/fire_reset.h"

namespace ai::environment {

FireReset::FireReset(std::unique_ptr<VirtualEnvironment> env)
    : env_(std::move(env)) {}

ScreenBuffer FireReset::reset() {
  ScreenBuffer observation;
  observation = env_->reset();
  auto fire_result = env_->step(ale::Action::PLAYER_A_FIRE);
  if (fire_result.terminated || fire_result.truncated) {
    observation = env_->reset();
  }
  fire_result = env_->step(ale::Action::PLAYER_A_UP);
  if (fire_result.terminated || fire_result.truncated) {
    observation = env_->reset();
  }
  return observation;
}

Step FireReset::step(const ale::Action &action) { return env_->step(action); }

ale::ALEInterface &FireReset::get_interface() { return env_->get_interface(); }

} // namespace ai::environment
