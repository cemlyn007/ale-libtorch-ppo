#pragma once
#include "ai/environment/environment.h"
#include <memory>
#include <random>

namespace ai::environment {

class NoopResetEnvironment : public VirtualEnvironment {
public:
  explicit NoopResetEnvironment(std::unique_ptr<VirtualEnvironment> env,
                                size_t max_noops, size_t seed);
  ScreenBuffer reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  std::unique_ptr<VirtualEnvironment> env_;
  int max_noops_;
  std::mt19937 random_generator_;
  std::uniform_int_distribution<size_t> distribution_;
};

} // namespace ai::environment
