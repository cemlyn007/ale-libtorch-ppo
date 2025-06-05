#include <ale/ale_interface.hpp>
#include <ale/version.hpp>
#include <iostream>

int main(int argc, char **argv) {
  auto path = argv[1];
  ale::ALEInterface ale;
  std::cout << "Initializing Arcade Learning Environment..." << std::endl;
  ale.loadROM(path);
  std::cout << "ALE initialized!" << std::endl;
  std::cout << "Number of available actions: "
            << ale.getMinimalActionSet().size() << std::endl;
  ale.reset_game();
  std::cout << "Game reset!" << std::endl;
  return 0;
}
