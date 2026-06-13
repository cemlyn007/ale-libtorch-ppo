#pragma once
#include <memory>

#include "ai/environment/environment.h"

namespace ai::environment {

class FireReset : public VirtualEnvironment {
 public:
  explicit FireReset(std::unique_ptr<VirtualEnvironment> env);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action &action, bool want_observation) override;
  ale::ALEInterface &get_interface() override;
  void serialize(std::ostream &os) override;
  void deserialize(std::istream &is) override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
};

}  // namespace ai::environment
