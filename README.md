# ALE-libtorch-PPO

<p align="center"><a href="https://youtu.be/MQsjzNbIrsQ"><img src="https://github.com/user-attachments/assets/f8b027b6-2294-4142-8fad-549f830d48a3" width="400"></a></p>

This project is a C++ application designed to train an agent to master Atari games, with a specific focus on the classic game "Breakout". It leverages reinforcement learning, implementing the Proximal Policy Optimization (PPO) algorithm to enable the agent to learn and improve its gameplay through trial and error.

Built using Bazel, this project integrates `libtorch` (the C++ frontend for PyTorch) for its neural network components and the Arcade Learning Environment (ALE) to interface with the Atari games. This combination provides a high-performance environment for cutting-edge AI research.

While Python-based libraries dominate the open-source RL scene by offering ease of use and a vast ecosystem, `ALE-libtorch-PPO` contributes a high-performance, C++-native alternative. It is designed for developers and researchers who need maximum performance, a clean C++ integration path, and a transparent, focused implementation of a state-of-the-art RL algorithm.

## Run Instructions
To run the project, follow these steps:

1. Install Bazel by following the [Bazel installation guide](https://bazel.build/install) for your operating system.

2. Clone the repository:
   ```bash
   git clone https://github.com/cemlyn007/ALE-libtorch-PPO.git
   cd ALE-libtorch-PPO
   ```

3. Download the ROMs:
   ```bash
   ./scripts/download_unpack_roms.sh
   ```

4. Train the agent using Bazel:
   ```bash
   bazel run //src/bin:train --compilation_mode=opt -- roms/breakout.bin logs/train video/train train configs/v0.yaml
   ```
   Or alternatively, with VS Code, you can run the tasks. The command line arguments do the following:
   1. Specify which ROM to use.
   2. Specify the directory to write TensorBoard logs to.
   3. Specify the directory to write videos to.
   4. Specify the group name used for logging parameters to TensorBoard.
   5. Specify the path to the YAML file containing the config to use for running the application.
   6. Optional: specify the location to write a libtorch profile to which can be examined using Perfetto.

## Contributions Welcome

We welcome contributions from the community! If you're interested in improving `ALE-libtorch-PPO`, here are some ways you can help:

*   **Reporting Bugs:** If you find a bug, please open an issue and provide as much detail as possible.
*   **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? We'd love to hear it.
*   **Code Contributions:** If you'd like to contribute code, please fork the repository and submit a pull request. We appreciate all contributions, from small bug fixes to major new features.

We look forward to collaborating with you!
