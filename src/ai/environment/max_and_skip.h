#pragma once
#include <memory>

#include "ai/environment/environment.h"

namespace ai::environment {

class MaxAndSkipEnvironment : public VirtualEnvironment {
 public:
  explicit MaxAndSkipEnvironment(std::unique_ptr<VirtualEnvironment> env,
                                 size_t skip);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action &action, bool want_observation) override;
  ale::ALEInterface &get_interface() override;
  void serialize(std::ostream &os) override;
  void deserialize(std::istream &is) override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
  size_t skip_;
  // Most recent frame that was grabbed but not emitted (the runner-up of the
  // max pool). Emitted when the episode ends before the pooling window so the
  // terminal step still carries a correctly-sized observation.
  ScreenBuffer last_frame_;
};

}  // namespace ai::environment
