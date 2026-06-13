#pragma once
#include <memory>
#include <random>

#include "ai/environment/environment.h"

namespace ai::environment {

class NoopResetEnvironment : public VirtualEnvironment {
 public:
  explicit NoopResetEnvironment(std::unique_ptr<VirtualEnvironment> env,
                                size_t max_noops, size_t seed);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action &action, bool want_observation) override;
  ale::ALEInterface &get_interface() override;
  void serialize(std::ostream &os) override;
  void deserialize(std::istream &is) override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
  std::mt19937 random_generator_;
  std::uniform_int_distribution<size_t> distribution_;
};

}  // namespace ai::environment
