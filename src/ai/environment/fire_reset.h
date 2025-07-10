#pragma once
#include "ai/environment/environment.h"
#include <memory>

namespace ai::environment {

class FireReset : public VirtualEnvironment {
public:
  explicit FireReset(std::unique_ptr<VirtualEnvironment> env);
  ScreenBuffer reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  std::unique_ptr<VirtualEnvironment> env_;
};

} // namespace ai::environment
