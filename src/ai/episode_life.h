#pragma once
#include "ai/environment.h"
#include <memory>

namespace ai::environment {

class EpisodeLife : public VEnvironment {
public:
  explicit EpisodeLife(std::unique_ptr<VEnvironment> env);
  void reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  std::unique_ptr<VEnvironment> env_;
  int lives_;
};

} // namespace ai::environment
