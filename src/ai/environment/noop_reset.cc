#include "ai/environment/noop_reset.h"

namespace ai::environment {

NoopResetEnvironment::NoopResetEnvironment(
    std::unique_ptr<VirtualEnvironment> env, size_t max_noops, size_t seed)
    : env_(std::move(env)),
      random_generator_(seed),
      // uniform_int_distribution is inclusive on both bounds.
      distribution_(1, max_noops) {}

ScreenBuffer NoopResetEnvironment::reset() {
  ScreenBuffer observation = env_->reset();
  size_t noops = distribution_(random_generator_);
  for (size_t i = 0; i < noops; ++i) {
    // Only the last noop's frame is returned, so earlier grabs are skipped.
    const bool last = i + 1 == noops;
    auto result = env_->step(ale::Action::PLAYER_A_NOOP, last);
    if (result.terminated || result.truncated) {
      observation = env_->reset();
    } else if (last) {
      observation = std::move(result.observation);
    }
  }
  return observation;
}

Step NoopResetEnvironment::step(const ale::Action &action,
                                bool want_observation) {
  return env_->step(action, want_observation);
}

ale::ALEInterface &NoopResetEnvironment::get_interface() {
  return env_->get_interface();
}

}  // namespace ai::environment
