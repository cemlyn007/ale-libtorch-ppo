#pragma once
#include <ale/ale_interface.hpp>
#include <filesystem>
#include <vector>

namespace ai::environment {

typedef std::vector<unsigned char> ScreenBuffer;

struct Step {
  ScreenBuffer observation;
  int reward;
  bool terminated;
  bool truncated;
  // Indicates if the game is completely over.
  // When true, terminated or truncated must also be true.
  bool game_over;
};

class VirtualEnvironment {
public:
  virtual ScreenBuffer reset() = 0;
  virtual Step step(const ale::Action &action) = 0;
  virtual ale::ALEInterface &get_interface() = 0;
};

class Environment : public VirtualEnvironment {
public:
  Environment(const std::filesystem::path &rom_path,
              size_t max_num_frames_per_episode, bool grayscale, int seed);
  ScreenBuffer reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  ale::ALEInterface ale_;
  const bool grayscale_;
  const size_t size_;
  ScreenBuffer get_observation();
};

} // namespace ai::environment