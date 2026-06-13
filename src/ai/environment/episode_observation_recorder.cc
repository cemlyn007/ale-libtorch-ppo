#include "ai/environment/episode_observation_recorder.h"

namespace ai::environment {

EpisodeObservationRecorder::EpisodeObservationRecorder(
    std::unique_ptr<VirtualEnvironment> env,
    const std::filesystem::path& video_path, size_t channels, size_t height,
    size_t width, size_t fps)
    : episode_index_(0),
      env_(std::move(env)),
      video_recorder_(video_path, channels, width, height, fps) {}

ScreenBuffer EpisodeObservationRecorder::reset() {
  ScreenBuffer observation = env_->reset();
  episode_index_++;
  std::filesystem::path path =
      "episode_" + std::to_string(episode_index_) + ".mp4";
  video_recorder_.open(path);
  video_recorder_.write(observation);
  return observation;
}

Step EpisodeObservationRecorder::step(const ale::Action& action,
                                      bool /*want_observation*/) {
  // The recording is made from the observation, so it is always wanted here
  // regardless of what the caller asked for.
  auto result = env_->step(action, true);
  video_recorder_.write(result.observation);
  if (result.terminated || result.truncated) video_recorder_.close();
  return result;
}

ale::ALEInterface& EpisodeObservationRecorder::get_interface() {
  return env_->get_interface();
}

// Video output is append-only and not part of restorable training state, so
// only the wrapped environment is snapshotted.
void EpisodeObservationRecorder::serialize(std::ostream& os) {
  env_->serialize(os);
}
void EpisodeObservationRecorder::deserialize(std::istream& is) {
  env_->deserialize(is);
}

}  // namespace ai::environment
