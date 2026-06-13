#pragma once
#include <memory>

#include "ai/environment/environment.h"

namespace ai::environment {

class ResizeEnvironment : public VirtualEnvironment {
 public:
  explicit ResizeEnvironment(std::unique_ptr<VirtualEnvironment> env,
                             int new_width, int new_height);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action &action, bool want_observation) override;
  ale::ALEInterface &get_interface() override;
  void serialize(std::ostream &os) override;
  void deserialize(std::istream &is) override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
  int width_;
  int height_;
  int new_width_;
  int new_height_;
  bool identity_;

  ScreenBuffer resize(const ScreenBuffer &observation) const;
};

}  // namespace ai::environment
