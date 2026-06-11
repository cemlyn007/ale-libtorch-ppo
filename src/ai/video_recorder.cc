#include "video_recorder.h"

#include <fcntl.h>
#include <sys/wait.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace ai::video_recorder {

namespace {
// popen runs the command through /bin/sh, so the output path must be
// single-quoted ('\'' splices an escaped quote between two quoted runs).
std::string shell_quote(const std::string& s) {
  std::string quoted = "'";
  for (char c : s) {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted += c;
  }
  quoted += "'";
  return quoted;
}
}  // namespace

VideoRecorder::VideoRecorder(const std::filesystem::path& video_dir,
                             size_t channels, size_t width, size_t height,
                             size_t fps)
    : video_dir_(video_dir),
      channels_(channels),
      width_(width),
      height_(height),
      fps_(fps) {
  if (channels_ == 1)
    pixel_format_ = "gray";
  else if (channels_ == 3)
    pixel_format_ = "rgb24";
  else if (channels_ == 4)
    pixel_format_ = "rgba";
  else
    throw std::runtime_error("Unsupported number of channels");
  // A missing directory would otherwise only surface as ffmpeg exiting after
  // popen already succeeded.
  std::filesystem::create_directories(video_dir_);
}

VideoRecorder::~VideoRecorder() {
  if (ffmpeg_stream_) {
    // Best-effort finalize: send EOF and reap ffmpeg so it writes the mp4
    // trailer. Never throw from a destructor (this runs during rollout
    // teardown).
    pclose(ffmpeg_stream_);
    ffmpeg_stream_ = nullptr;
  }
}

void VideoRecorder::open(const std::filesystem::path& path) {
  if (ffmpeg_stream_)
    throw std::runtime_error("Video recording has already started");
  std::string command = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt " +
                        pixel_format_ + " -s " + std::to_string(width_) + "x" +
                        std::to_string(height_) + " -r " +
                        std::to_string(fps_) +
                        // veryfast/-threads 1: diagnostics video must not
                        // steal cores from the CPU-bound env threads.
                        " -i - -c:v libx264 -preset veryfast -crf 28 "
                        "-threads 1 -pix_fmt yuv420p "
                        "-movflags +faststart -hide_banner -loglevel error " +
                        shell_quote((video_dir_ / path).string());
  ffmpeg_stream_ = popen(command.c_str(), "w");
  if (!ffmpeg_stream_)
    throw std::runtime_error(std::string("Failed to open pipe for ffmpeg: ") +
                             std::strerror(errno));
  // A full-res RGB frame (~100KB) exceeds the default 64KB pipe, so every
  // write would block until x264 drains. 1MB buys ~10 frames of slack;
  // best-effort, failure just keeps the default capacity.
  fcntl(fileno(ffmpeg_stream_), F_SETPIPE_SZ, 1 << 20);
}

void VideoRecorder::write(std::span<const unsigned char> frame) {
  if (!ffmpeg_stream_)
    throw std::runtime_error("Video recording has not been started");
  size_t expected = width_ * height_ * channels_;
  if (frame.size() != expected)
    throw std::runtime_error("Frame size mismatch: got " +
                             std::to_string(frame.size()) +
                             " bytes, expected " + std::to_string(expected));
  // SIGPIPE is ignored process-wide, so a dead ffmpeg shows up here as a
  // short write rather than killing the training run.
  if (fwrite(frame.data(), 1, frame.size(), ffmpeg_stream_) != frame.size())
    throw std::runtime_error(
        "Failed to write frame to ffmpeg (it likely "
        "exited; check disk space and the output path)");
}

void VideoRecorder::close() {
  if (!ffmpeg_stream_)
    throw std::runtime_error("Video recording has not been started");
  int status = pclose(ffmpeg_stream_);
  // Null out before any throw so the destructor cannot pclose a dead stream.
  ffmpeg_stream_ = nullptr;
  if (status == -1)
    throw std::runtime_error(std::string("pclose() failed: ") +
                             std::strerror(errno));
  if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
    throw std::runtime_error("ffmpeg exited with status " +
                             std::to_string(WEXITSTATUS(status)));
  if (WIFSIGNALED(status))
    throw std::runtime_error("ffmpeg killed by signal " +
                             std::to_string(WTERMSIG(status)));
}

}  // namespace ai::video_recorder
