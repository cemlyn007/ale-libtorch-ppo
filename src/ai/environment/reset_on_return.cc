#include "ai/environment/reset_on_return.h"

namespace ai::environment {

ResetOnReturnEnvironment::ResetOnReturnEnvironment(
    std::unique_ptr<VirtualEnvironment> env, ale::reward_t max_return)
    : env_(std::move(env)), max_return_(max_return), current_return_(0) {}

ScreenBuffer ResetOnReturnEnvironment::reset() {
  current_return_ = 0;
  return env_->reset();
}

Step ResetOnReturnEnvironment::step(const ale::Action &action) {
  Step step = env_->step(action);
  current_return_ += step.reward;
  if (current_return_ >= max_return_) {
    step.truncated = true;
    step.game_over = true;
  }
  return step;
}

ale::ALEInterface &ResetOnReturnEnvironment::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
