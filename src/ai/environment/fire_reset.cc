#include "ai/environment/fire_reset.h"

namespace ai::environment {

FireReset::FireReset(std::unique_ptr<VirtualEnvironment> env)
    : env_(std::move(env)) {}

ScreenBuffer FireReset::reset() {
  // Match Gym's FireResetEnv: the episode starts on the frame after FIRE+UP,
  // so that is the observation to return (the reset and FIRE frames' pixels
  // are never seen by the agent).
  env_->reset();
  auto fire_result =
      env_->step(ale::Action::PLAYER_A_FIRE, /*want_observation=*/false);
  if (fire_result.terminated || fire_result.truncated) env_->reset();
  fire_result = env_->step(ale::Action::PLAYER_A_UP);
  if (fire_result.terminated || fire_result.truncated) return env_->reset();
  return std::move(fire_result.observation);
}

Step FireReset::step(const ale::Action &action, bool want_observation) {
  return env_->step(action, want_observation);
}

ale::ALEInterface &FireReset::get_interface() { return env_->get_interface(); }

void FireReset::serialize(std::ostream &os) { env_->serialize(os); }
void FireReset::deserialize(std::istream &is) { env_->deserialize(is); }

}  // namespace ai::environment
