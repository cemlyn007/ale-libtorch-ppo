#pragma once
#include <cstdio>
#include <filesystem>
#include <span>

namespace ai::video_recorder {

class VideoRecorder {
 public:
  VideoRecorder(const std::filesystem::path& video_dir, size_t channels,
                size_t width, size_t height, size_t fps = 30);
  ~VideoRecorder();
  // Owns a raw FILE*; copying or moving would double-pclose.
  VideoRecorder(const VideoRecorder&) = delete;
  VideoRecorder& operator=(const VideoRecorder&) = delete;
  VideoRecorder(VideoRecorder&&) = delete;
  VideoRecorder& operator=(VideoRecorder&&) = delete;

  void open(const std::filesystem::path& path);
  void write(std::span<const unsigned char> frame);
  void close();

 private:
  std::filesystem::path video_dir_;
  size_t channels_;
  size_t width_;
  size_t height_;
  size_t fps_;
  std::string pixel_format_;
  std::FILE* ffmpeg_stream_ = nullptr;
};
}  // namespace ai::video_recorder
