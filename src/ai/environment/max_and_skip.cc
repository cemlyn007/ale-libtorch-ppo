#include "ai/environment/max_and_skip.h"

namespace ai::environment {

MaxAndSkipEnvironment::MaxAndSkipEnvironment(
    std::unique_ptr<VirtualEnvironment> env, size_t skip)
    : env_(std::move(env)), skip_(skip) {}

ScreenBuffer MaxAndSkipEnvironment::reset() { return env_->reset(); }

Step MaxAndSkipEnvironment::step(const ale::Action &action) {
  Step result = {
      .reward = 0,
      .terminated = false,
      .truncated = false,
  };
  ScreenBuffer prev_observation;
  ScreenBuffer curr_observation;
  for (size_t i = 0; i < skip_; ++i) {
    auto step_result = env_->step(action);
    result.reward += step_result.reward;
    result.terminated |= step_result.terminated;
    result.truncated |= step_result.truncated;
    prev_observation = std::move(curr_observation);
    curr_observation = step_result.observation;
    if (result.terminated || result.truncated) {
      break;
    }
  }
  if (!prev_observation.empty() &&
      prev_observation.size() == curr_observation.size()) {
    ScreenBuffer pooled(curr_observation.size());
    for (size_t i = 0; i < curr_observation.size(); ++i) {
      pooled[i] = std::max(prev_observation[i], curr_observation[i]);
    }
    result.observation = std::move(pooled);
  } else {
    result.observation = std::move(curr_observation);
  }
  return result;
}

ale::ALEInterface &MaxAndSkipEnvironment::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
