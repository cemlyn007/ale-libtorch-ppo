#pragma once
#include <memory>

#include "ai/environment/environment.h"
#include "ai/video_recorder.h"

namespace ai::environment {

class EpisodeRecorder : public VirtualEnvironment {
 public:
  explicit EpisodeRecorder(std::unique_ptr<VirtualEnvironment> env,
                           const std::filesystem::path &video_path,
                           bool grayscale);
  ScreenBuffer reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

 private:
  std::unique_ptr<VirtualEnvironment> env_;
  bool grayscale_;
  size_t episode_index_;
  std::vector<unsigned char> buffer_;
  ai::video_recorder::VideoRecorder video_recorder_;

  void update_buffer();
};

}  // namespace ai::environment
