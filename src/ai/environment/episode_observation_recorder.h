#pragma once
#include <memory>

#include "ai/environment/environment.h"
#include "ai/video_recorder.h"

namespace ai::environment {

class EpisodeObservationRecorder : public VirtualEnvironment {
 public:
  explicit EpisodeObservationRecorder(std::unique_ptr<VirtualEnvironment> env,
                                      const std::filesystem::path& video_path,
                                      size_t channels, size_t height,
                                      size_t width, size_t fps);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action& action, bool want_observation) override;
  ale::ALEInterface& get_interface() override;
  void serialize(std::ostream& os) override;
  void deserialize(std::istream& is) override;

 private:
  size_t episode_index_;
  std::unique_ptr<VirtualEnvironment> env_;
  ai::video_recorder::VideoRecorder video_recorder_;
};

}  // namespace ai::environment
