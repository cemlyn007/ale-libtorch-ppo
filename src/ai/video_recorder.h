#include <filesystem>
#include <vector>

namespace ai::video_recorder {

class VideoRecorder {
public:
  VideoRecorder(const std::filesystem::path &, size_t channels, size_t width,
                size_t height, size_t fps = 30);

  void add(const unsigned char *data);

  void complete(std::filesystem::path &);

private:
  std::string video_path_;
  size_t channels_;
  size_t width_;
  size_t height_;
  size_t fps_;
  size_t frame_index_;
  std::vector<unsigned char> frames_;
};
} // namespace ai::video_recorder