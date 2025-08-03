#include "video_recorder.h"
#include <filesystem>
#include <stb_image_write.h>
#include <stdio.h>
namespace ai::video_recorder {

VideoRecorder::VideoRecorder(const std::filesystem::path &video_path,
                             size_t channels, size_t width, size_t height,
                             size_t fps)
    : video_path_(video_path), channels_(channels), width_(width),
      height_(height), fps_(fps) {
  if (channels_ == 1)
    pixel_format_ = "gray";
  else if (channels_ == 3)
    pixel_format_ = "rgb24";
  else if (channels_ == 4)
    pixel_format_ = "rgba";
  else
    throw std::runtime_error("Unsupported number of channels");
}

void VideoRecorder::add(const unsigned char *data) {
  size_t size = width_ * height_ * channels_;
  frames_.emplace_back(data, data + size);
}

void VideoRecorder::complete(std::filesystem::path &path) {
  if (frames_.empty())
    throw std::runtime_error("No frames to write to video");
  std::string command = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt " +
                        pixel_format_ + " -s " + std::to_string(width_) + "x" +
                        std::to_string(height_) + " -r " +
                        std::to_string(fps_) +
                        " -i - -c:v libx264 -pix_fmt yuv420p -hide_banner "
                        "-loglevel error " +
                        (video_path_ / path).string();
  auto stream = popen(command.data(), "w");
  if (!stream)
    throw std::runtime_error("Failed to open pipe for ffmpeg");
  for (const auto &frame : frames_)
    fwrite(frame.data(), sizeof(unsigned char), frame.size(), stream);
  int status = pclose(stream);
  if (status == -1)
    throw std::runtime_error("Error reported by pclose()");
  frames_.clear();
}

} // namespace ai::video_recorder