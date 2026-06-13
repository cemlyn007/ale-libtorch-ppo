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
  virtual ~VirtualEnvironment() = default;
  virtual ScreenBuffer reset() = 0;
  // want_observation=false lets implementations skip the screen grab (and any
  // image work built on it) for frames whose pixels the caller will discard;
  // the returned Step then carries an empty observation.
  virtual Step step(const ale::Action &action, bool want_observation) = 0;
  Step step(const ale::Action &action) { return step(action, true); }
  virtual ale::ALEInterface &get_interface() = 0;
};

class Environment : public VirtualEnvironment {
 public:
  Environment(const std::filesystem::path &rom_path,
              size_t max_num_frames_per_episode, bool grayscale, int seed);
  using VirtualEnvironment::step;
  ScreenBuffer reset() override;
  Step step(const ale::Action &action, bool want_observation) override;
  ale::ALEInterface &get_interface() override;

 private:
  ale::ALEInterface ale_;
  const bool grayscale_;
  size_t size_;
  ScreenBuffer get_observation();
};

}  // namespace ai::environment