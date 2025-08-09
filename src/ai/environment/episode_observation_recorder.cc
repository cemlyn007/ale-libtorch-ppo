#include "ai/environment/episode_observation_recorder.h"

namespace ai::environment {

EpisodeObservationRecorder::EpisodeObservationRecorder(
    std::unique_ptr<VirtualEnvironment> env,
    const std::filesystem::path &video_path, size_t channels, size_t height,
    size_t width)
    : episode_index_(0), env_(std::move(env)),
      video_recorder_(video_path, channels, width, height, 30) {}

ScreenBuffer EpisodeObservationRecorder::reset() {
  ScreenBuffer observation = env_->reset();
  episode_index_++;
  std::filesystem::path path =
      "episode_" + std::to_string(episode_index_) + ".mp4";
  video_recorder_.open(path);
  video_recorder_.write(observation.data());
  return observation;
}

Step EpisodeObservationRecorder::step(const ale::Action &action) {
  auto result = env_->step(action);
  video_recorder_.write(result.observation.data());
  if (result.terminated || result.truncated)
    video_recorder_.close();
  return result;
}

ale::ALEInterface &EpisodeObservationRecorder::get_interface() {
  return env_->get_interface();
}

} // namespace ai::environment
