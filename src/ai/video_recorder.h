#pragma once
#include <filesystem>
#include <stdio.h>

namespace ai::video_recorder {

class VideoRecorder {
public:
  VideoRecorder(const std::filesystem::path &, size_t channels, size_t width,
                size_t height, size_t fps = 30);
  ~VideoRecorder();

  void open(const std::filesystem::path &path);
  void write(const unsigned char *data);
  void close();

private:
  std::filesystem::path video_path_;
  size_t channels_;
  size_t width_;
  size_t height_;
  size_t fps_;
  std::string pixel_format_;
  FILE *ffmpeg_stream_ = nullptr;
};
} // namespace ai::video_recorder