#include "ai/environment/truncate_on_episode_return.h"

namespace ai::environment {

TruncateOnEpisodeReturnEnvironment::TruncateOnEpisodeReturnEnvironment(
    std::unique_ptr<VirtualEnvironment> env, ale::reward_t max_return)
    : env_(std::move(env)), max_return_(max_return), current_return_(0) {}

ScreenBuffer TruncateOnEpisodeReturnEnvironment::reset() {
  current_return_ = 0;
  return env_->reset();
}

Step TruncateOnEpisodeReturnEnvironment::step(const ale::Action &action) {
  if (current_return_ >= max_return_)
    throw std::runtime_error(
        "Cannot step, current return has reached or exceeded max return.");
  Step step = env_->step(action);
  current_return_ += step.reward;
  if (!step.terminated && current_return_ >= max_return_) {
    step.truncated = true;
    step.game_over = true;
  }
  return step;
}

ale::ALEInterface &TruncateOnEpisodeReturnEnvironment::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
