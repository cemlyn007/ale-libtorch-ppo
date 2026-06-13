#include "ai/environment/noop_reset.h"

#include <sstream>

#include "ai/environment/state_io.h"

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

void NoopResetEnvironment::serialize(std::ostream &os) {
  // mt19937 round-trips via its stream operators.
  std::ostringstream rng;
  rng << random_generator_;
  state_io::write_bytes(os, rng.str());
  env_->serialize(os);
}
void NoopResetEnvironment::deserialize(std::istream &is) {
  std::istringstream rng(state_io::read_bytes(is));
  rng >> random_generator_;
  env_->deserialize(is);
}

}  // namespace ai::environment
