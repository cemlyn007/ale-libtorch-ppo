#pragma once
#include "ai/environment/environment.h"
#include <memory>

namespace ai::environment {

class EpisodeLife : public VirtualEnvironment {
public:
  explicit EpisodeLife(std::unique_ptr<VirtualEnvironment> env);
  void reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  std::unique_ptr<VirtualEnvironment> env_;
  int lives_;
};

} // namespace ai::environment
