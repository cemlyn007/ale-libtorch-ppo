#include "ai/environment/max_and_skip.h"

#include "ai/environment/state_io.h"

namespace ai::environment {

MaxAndSkipEnvironment::MaxAndSkipEnvironment(
    std::unique_ptr<VirtualEnvironment> env, size_t skip)
    : env_(std::move(env)), skip_(skip) {
  if (!env_) throw std::invalid_argument("Environment must not be null.");
  if (skip_ == 0) throw std::invalid_argument("Skip must be greater than 0.");
}

ScreenBuffer MaxAndSkipEnvironment::reset() {
  ScreenBuffer observation = env_->reset();
  // Seed the terminal-step fallback so an episode that ends before the first
  // pooling window still emits a frame.
  last_frame_ = observation;
  return observation;
}

Step MaxAndSkipEnvironment::step(const ale::Action &action,
                                 bool want_observation) {
  // Accumulate rewards over skipped frames.
  ale::reward_t total_reward = 0;
  Step result;

  ScreenBuffer prev_observation;
  ScreenBuffer curr_observation;
  for (size_t i = 0; i < skip_; ++i) {
    // Only the last two frames can survive the max pool, so the earlier
    // frames' screens are never grabbed.
    const bool want = want_observation && i + 2 >= skip_;
    result = env_->step(action, want);
    total_reward += result.reward;
    if (want) {
      prev_observation = std::move(curr_observation);
      curr_observation = std::move(result.observation);
    }
    if (result.terminated || result.truncated) {
      break;
    }
  }
  if (!prev_observation.empty() &&
      prev_observation.size() == curr_observation.size()) {
    for (size_t i = 0; i < curr_observation.size(); ++i) {
      curr_observation[i] = std::max(prev_observation[i], curr_observation[i]);
    }
  }
  if (want_observation && curr_observation.empty()) {
    // Episode ended before the pooling window; fall back to the most recent
    // real frame (Gymnasium's AtariPreprocessing emits stale pixels here too).
    curr_observation = last_frame_;
  } else if (!prev_observation.empty()) {
    last_frame_ = std::move(prev_observation);
  }
  result.observation = std::move(curr_observation);
  result.reward = total_reward;
  return result;
}

ale::ALEInterface &MaxAndSkipEnvironment::get_interface() {
  return env_->get_interface();
}

void MaxAndSkipEnvironment::serialize(std::ostream &os) {
  state_io::write_vector(os, last_frame_);
  env_->serialize(os);
}
void MaxAndSkipEnvironment::deserialize(std::istream &is) {
  last_frame_ = state_io::read_vector<unsigned char>(is);
  env_->deserialize(is);
}

}  // namespace ai::environment
