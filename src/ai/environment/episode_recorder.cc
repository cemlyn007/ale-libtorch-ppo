#include "ai/environment/episode_recorder.h"

namespace ai::environment {

EpisodeRecorder::EpisodeRecorder(std::unique_ptr<VirtualEnvironment> env,
                                 const std::filesystem::path &video_path,
                                 bool grayscale)
    : env_(std::move(env)), grayscale_(grayscale), episode_index_(0),
      buffer_([&]() {
        auto screen = env_->get_interface().getScreen();
        if (grayscale) {
          return std::vector<unsigned char>(screen.height() * screen.width());
        } else {
          return std::vector<unsigned char>(3 * screen.height() *
                                            screen.width());
        }
      }()),
      video_recorder_([&]() {
        auto screen = env_->get_interface().getScreen();
        return ai::video_recorder::VideoRecorder(
            video_path, grayscale ? 1 : 3, screen.width(), screen.height(), 30);
      }()) {}

ScreenBuffer EpisodeRecorder::reset() {
  ScreenBuffer observation = env_->reset();
  update_buffer();
  episode_index_++;
  std::filesystem::path path =
      "episode_" + std::to_string(episode_index_) + ".mp4";
  video_recorder_.open(path);
  video_recorder_.write(buffer_.data());
  return observation;
}

Step EpisodeRecorder::step(const ale::Action &action) {
  auto result = env_->step(action);
  update_buffer();
  video_recorder_.write(buffer_.data());
  if (result.terminated || result.truncated)
    video_recorder_.close();
  return result;
}

ale::ALEInterface &EpisodeRecorder::get_interface() {
  return env_->get_interface();
}

void EpisodeRecorder::update_buffer() {
  if (grayscale_)
    env_->get_interface().getScreenGrayscale(buffer_);
  else
    env_->get_interface().getScreenRGB(buffer_);
}

} // namespace ai::environment
