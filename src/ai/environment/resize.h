#pragma once
#include "ai/environment/environment.h"
#include <memory>

namespace ai::environment {

class ResizeEnvironment : public VirtualEnvironment {
public:
  explicit ResizeEnvironment(std::unique_ptr<VirtualEnvironment> env,
                             int new_width, int new_height);
  ScreenBuffer reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  std::unique_ptr<VirtualEnvironment> env_;
  int width_;
  int height_;
  int new_width_;
  int new_height_;

  ScreenBuffer resize(const ScreenBuffer &observation) const;
};

} // namespace ai::environment
