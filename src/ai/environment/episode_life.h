#pragma once
#include <memory>

#include "ai/environment/environment.h"

namespace ai::environment {

class EpisodeLife : public VirtualEnvironment {
 public:
  explicit EpisodeLife(std::unique_ptr<VirtualEnvironment> env);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action &action, bool want_observation) override;
  ale::ALEInterface &get_interface() override;
  void serialize(std::ostream &os) override;
  void deserialize(std::istream &is) override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
  int lives_;
  bool game_over_;
};

}  // namespace ai::environment
