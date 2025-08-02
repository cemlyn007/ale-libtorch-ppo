#include "ai/environment/max_and_skip.h"

namespace ai::environment {

MaxAndSkipEnvironment::MaxAndSkipEnvironment(
    std::unique_ptr<VirtualEnvironment> env, size_t skip)
    : env_(std::move(env)), skip_(skip) {
  if (!env_)
    throw std::invalid_argument("Environment must not be null.");
  if (skip_ == 0)
    throw std::invalid_argument("Skip must be greater than 0.");
}

ScreenBuffer MaxAndSkipEnvironment::reset() { return env_->reset(); }

Step MaxAndSkipEnvironment::step(const ale::Action &action) {
  // Accumulate rewards over skipped frames.
  ale::reward_t total_reward = 0;
  Step result;

  ScreenBuffer prev_observation;
  ScreenBuffer curr_observation;
  for (size_t i = 0; i < skip_; ++i) {
    result = env_->step(action);
    total_reward += result.reward;

    prev_observation = std::move(curr_observation);
    curr_observation = std::move(result.observation);
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
  result.reward = total_reward;
  return result;
}

ale::ALEInterface &MaxAndSkipEnvironment::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
