#pragma once
#include <ale/ale_interface.hpp>
#include <memory>

#include "ai/environment/environment.h"

namespace ai::environment {

class TruncateOnEpisodeReturnEnvironment : public VirtualEnvironment {
 public:
  explicit TruncateOnEpisodeReturnEnvironment(
      std::unique_ptr<VirtualEnvironment> env, ale::reward_t max_return);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action &action, bool want_observation) override;
  ale::ALEInterface &get_interface() override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
  ale::reward_t max_return_;
  ale::reward_t current_return_;
};

}  // namespace ai::environment
