#pragma once
#include <memory>

#include "ai/environment/environment.h"

namespace ai::environment {

class MaxAndSkipEnvironment : public VirtualEnvironment {
 public:
  explicit MaxAndSkipEnvironment(std::unique_ptr<VirtualEnvironment> env,
                                 size_t skip);
  ScreenBuffer reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
  size_t skip_;
};

}  // namespace ai::environment
