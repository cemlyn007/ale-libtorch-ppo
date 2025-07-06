#include "video_recorder.h"
#include <filesystem>
#include <stb_image_write.h>
#include <stdio.h>

namespace ai::video_recorder {

VideoRecorder::VideoRecorder(const std::filesystem::path &video_path,
                             size_t channels, size_t width, size_t height,
                             size_t fps)
    : video_path_(video_path), channels_(channels), width_(width),
      height_(height), fps_(fps) {}

void VideoRecorder::add(const unsigned char *data) {
  int length;
  auto frame_pointer = stbi_write_png_to_mem(data, channels_ * width_, width_,
                                             height_, channels_, &length);
  if (!frame_pointer) {
    throw std::runtime_error("Failed to write frame to memory");
  }
  std::vector<unsigned char> frame(frame_pointer, frame_pointer + length);
  free(frame_pointer);
  frames_.reserve(frames_.size() + frame.size());
  frames_.insert(frames_.end(), frame.begin(), frame.end());
}

void VideoRecorder::complete(std::filesystem::path &path) {
  if (frames_.empty()) {
    throw std::runtime_error("No frames to write to video");
  }
  std::string command = "ffmpeg -framerate " + std::to_string(fps_) +
                        " -hide_banner -loglevel error -f image2pipe -c:v png "
                        "-i - -c:v libx264 -y " +
                        path.string();
  auto stream = popen(command.data(), "w");
  if (!stream) {
    throw std::runtime_error("Failed to open pipe for ffmpeg");
  }
  fwrite(frames_.data(), sizeof(unsigned char), frames_.size(), stream);
  pclose(stream);
  frames_.clear();
}

} // namespace ai::video_recorder