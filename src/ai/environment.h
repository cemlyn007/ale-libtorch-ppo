#pragma once
#include <ale/ale_interface.hpp>
#include <filesystem>

namespace ai::environment {

struct Step {
  int reward;
  bool terminated;
  bool truncated;
};

class VEnvironment {
public:
  virtual void reset() = 0;
  virtual Step step(const ale::Action &action) = 0;
  virtual ale::ALEInterface &get_interface() = 0;
};

class Environment : public VEnvironment {
public:
  Environment(const std::filesystem::path &rom_path,
              size_t max_num_frames_per_episode, size_t frame_skip,
              float repeat_action_probability, int seed);
  void reset() override;
  Step step(const ale::Action &action) override;
  ale::ALEInterface &get_interface() override;

private:
  ale::ALEInterface ale_;
};

} // namespace ai::environment