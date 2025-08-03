#include "video_recorder.h"
#include <filesystem>
#include <stdexcept>

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

VideoRecorder::~VideoRecorder() {
  if (ffmpeg_stream_) {
    close();
  }
}

void VideoRecorder::open(const std::filesystem::path &path) {
  if (ffmpeg_stream_)
    throw std::runtime_error("Video recording has already started");
  std::string command = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt " +
                        pixel_format_ + " -s " + std::to_string(width_) + "x" +
                        std::to_string(height_) + " -r " +
                        std::to_string(fps_) +
                        " -i - -c:v libx264 -pix_fmt yuv420p -hide_banner "
                        "-loglevel error " +
                        (video_path_ / path).string();
  ffmpeg_stream_ = popen(command.data(), "w");
  if (!ffmpeg_stream_)
    throw std::runtime_error("Failed to open pipe for ffmpeg");
}

void VideoRecorder::write(const unsigned char *data) {
  if (!ffmpeg_stream_)
    throw std::runtime_error("Video recording has not been started");
  size_t size = width_ * height_ * channels_;
  fwrite(data, sizeof(unsigned char), size, ffmpeg_stream_);
}

void VideoRecorder::close() {
  if (!ffmpeg_stream_)
    throw std::runtime_error("Video recording has not been started");
  int status = pclose(ffmpeg_stream_);
  if (status == -1) {
    throw std::runtime_error("Error reported by pclose()");
  }
  ffmpeg_stream_ = nullptr;
}

} // namespace ai::video_recorder