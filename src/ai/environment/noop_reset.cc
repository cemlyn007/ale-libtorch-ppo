#include "ai/environment/noop_reset.h"

namespace ai::environment {

NoopResetEnvironment::NoopResetEnvironment(
    std::unique_ptr<VirtualEnvironment> env, size_t max_noops, size_t seed)
    : env_(std::move(env)), random_generator_(seed),
      distribution_(1, max_noops + 1) {}

ScreenBuffer NoopResetEnvironment::reset() {
  ScreenBuffer observation = env_->reset();
  size_t noops = distribution_(random_generator_);
  for (size_t i = 0; i < noops; ++i) {
    auto result = env_->step(ale::Action::PLAYER_A_NOOP);
    if (result.terminated || result.truncated) {
      observation = env_->reset();
    } else {
      observation = result.observation;
    }
  }
  return observation;
}

Step NoopResetEnvironment::step(const ale::Action &action) {
  return env_->step(action);
}

ale::ALEInterface &NoopResetEnvironment::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
