#include "ale/ale_interface.hpp"
#include <iostream>

int main() {
  ale::ALEInterface ale;

  // Set ROM file
  std::cout << "Initializing Arcade Learning Environment..." << std::endl;
  ale.loadROM(
      "/home/cemlyn/Development/ale-bazel/roms/breakout.bin"); // You'll need to
                                                               // provide an
                                                               // Atari ROM file

  std::cout << "ALE initialized!" << std::endl;
  std::cout << "Number of available actions: "
            << ale.getMinimalActionSet().size() << std::endl;

  return 0;
}
