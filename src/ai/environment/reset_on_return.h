#pragma once
#include "ai/environment/environment.h"
#include <ale/ale_interface.hpp>
#include <memory>

namespace ai::environment {

class ResetOnReturnEnvironment : public VirtualEnvironment {
public:
  explicit ResetOnReturnEnvironment(std::unique_ptr<VirtualEnvironment> env,
                                    ale::reward_t max_return);
  ScreenBuffer reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  std::unique_ptr<VirtualEnvironment> env_;
  ale::reward_t max_return_;
  ale::reward_t current_return_;
};

} // namespace ai::environment
