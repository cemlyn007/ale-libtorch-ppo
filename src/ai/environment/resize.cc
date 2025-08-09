#include "resize.h"
#include "ai/vision.h"
#include <stdexcept>

namespace ai::environment {

ResizeEnvironment::ResizeEnvironment(std::unique_ptr<VirtualEnvironment> env,
                                     int new_width, int new_height)
    : env_(std::move(env)),
      width_([&] { return env_->get_interface().getScreen().width(); }()),
      height_([&] { return env_->get_interface().getScreen().height(); }()),
      new_width_(new_width), new_height_(new_height) {
  if (!env_)
    throw std::invalid_argument("Environment must not be null.");
  if (new_width_ <= 0 || new_height_ <= 0)
    throw std::invalid_argument("new_width and new_height must be > 0");
}

ScreenBuffer ResizeEnvironment::reset() {
  auto observation = env_->reset();
  return resize(observation);
}

Step ResizeEnvironment::step(const ale::Action &action) {
  auto result = env_->step(action);
  result.observation = resize(result.observation);
  return result;
}

ale::ALEInterface &ResizeEnvironment::get_interface() {
  return env_->get_interface();
}

ScreenBuffer ResizeEnvironment::resize(const ScreenBuffer &observation) const {
  // Expect grayscale input of size height_ * width_
  if (static_cast<int>(observation.size()) != width_ * height_)
    throw std::invalid_argument("ResizeEnvironment expects grayscale "
                                "observation with size width*height");
  return ai::vision::resize_grayscale_image(observation, width_, height_,
                                            new_width_, new_height_);
}

} // namespace ai::environment
