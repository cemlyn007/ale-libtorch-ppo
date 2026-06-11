# ALE-libtorch-PPO

<p align="center"><a href="https://youtu.be/MQsjzNbIrsQ"><img src="https://github.com/user-attachments/assets/f8b027b6-2294-4142-8fad-549f830d48a3" width="400"></a></p>

This project is a C++ application designed to train an agent to master Atari games, with a specific focus on the classic game "Breakout". It leverages reinforcement learning, implementing the Proximal Policy Optimisation (PPO) algorithm to enable the agent to learn and improve its gameplay through trial and error. Multi-threading is used to achieve greater throughput for interacting with the game environment in parallel.

Built using Bazel, this project integrates `libtorch` (the C++ frontend for PyTorch) for its neural network components and the Arcade Learning Environment (ALE) to interface with the Atari games. This combination provides a high-performance environment for cutting-edge AI research.

While Python-based libraries dominate the open-source RL scene by offering ease of use and a vast ecosystem, `ALE-libtorch-PPO` contributes a high-performance, C++-native alternative. It is designed for developers and researchers who need high performance, a clean C++ integration path, and a transparent, focused implementation of a strong & popular RL algorithm.

## Run Instructions
To run the project, follow these steps:

1. Install Bazel by following the [Bazel installation guide](https://bazel.build/install) for your operating system.

2. Install FFmpeg which the environment video recorder uses to generate MP4s of the agent playing the game.

3. Clone the repository:
   ```shell
   git clone https://github.com/cemlyn007/ALE-libtorch-PPO.git
   cd ALE-libtorch-PPO
   ```

4. Download the ROMs:
   ```shell
   mkdir roms
   ./scripts/download_unpack_roms.sh
   ```

5. Train the agent using Bazel:
   ```shell
   bazel run //src/bin:train --compilation_mode=opt -- --rom $(pwd)/roms/breakout.bin --log-path $(pwd)/logs/train --video-dir $(pwd)/videos/train --group train --config $(pwd)/configs/v0.yaml
   ```
   Or alternatively, with VS Code, you can run the tasks. Pass `--help` to list every flag. The command line options do the following:
   - `--rom`: Path to the Atari ROM to train on.
   - `--log-path`: TensorBoard log path prefix; `.tfevents.<timestamp>` is appended at runtime.
   - `--video-dir`: Directory to write videos to (required when `record_video` is set in the config).
   - `--group`: Group name used when logging parameters to TensorBoard.
   - `--config`: Path to the YAML file containing the config to use for running the application.
   - `--profile`: Optional path to write a libtorch profile to, which can be examined using Perfetto.

### Checkpointing

Each run is a self-contained directory — `<base>.<start_time>/` — holding its TensorBoard event file and its checkpoints (`latest.pt`, `best.pt`). Two config keys control checkpointing (see any file in `configs/`):

* `checkpoint_interval`: write `latest.pt` every N rollouts, and `best.pt` whenever the mean episode return improves. `latest.pt` is also flushed on a graceful stop (Ctrl-C / `SIGTERM`) and when a run finishes between intervals, so the newest completed rollout is never lost. `0` disables checkpointing entirely.
* `resume_from`: path to a checkpoint `.pt` to restore from. Empty starts fresh.

A checkpoint bundles the network weights, optimizer (Adam) state, the next rollout index (so the learning-rate schedule continues), and the global env step. Each save also drops a `checkpoint` text marker on the TensorBoard timeline so saves and resume points are visible there.

**Resuming continues the same TensorBoard experiment.** Because `resume_from` points at a checkpoint inside an existing run directory, the resumed process writes its new event file into that *same* directory, and TensorBoard merges all event files in a directory into one run. The restored global step means the metric curves continue from where they stopped rather than restarting at 0 — so an interrupted run picks up as a single, continuous timeline. The rollout RNG and environment state are not saved, so resumption is approximate rather than bit-exact.

## Results
Evaluated using the following hardware:
* ASUS ROG STRIX X670E-F GAMING WIFI
* AMD Ryzen™ 9 7950X3D × 32
* NVIDIA GeForce RTX™ 4090

### V0 Config
<p align="center"><img width="400" alt="TensorBoard showing PPO achieving score of 400 on Breakout within 10 million agent steps." src="https://github.com/user-attachments/assets/e1b62320-87e8-4ea9-9dd7-dbed2d0dfe98" /></p>
Achieved 10 million agent steps in 37 minutes and 39 seconds, using the v0 config, with an average steps per second of ~4426 with video recording enabled as well.

### V1 Config
<p align="center"><img width="400" alt="TensorBoard showing PPO achieving the maximum score on Breakout of 864." src="https://github.com/user-attachments/assets/d1bcec5b-20a2-49f6-ad9a-387f81950cff" /></p>
Achieved an average of ~26,289 steps per second, with video recording enabled, with hardware still not fully utilised.

## Profiling
There are three views for profiling this application, using `./scripts/flamegraph.sh`, running the application with a 6th command line argument which specifies where to save the Perfetto profile, lastly you can use nsys to profile the application. The flamegraph script will generate a flamegraph of the application, which can be viewed in a web browser. The Perfetto profile can be opened in the Perfetto UI, and NVIDIA Nsight Systems UI can also be used for profiling if you hook up the path to the `train` binary.

## Contributions Welcome

I welcome contributions from the community! If you're interested in improving `ALE-libtorch-PPO`, here are some ways you can help:

*   **Reporting Bugs:** If you find a bug, please open an issue and provide as much detail as possible.
*   **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? I'd love to hear it.
*   **Code Contributions:** If you'd like to contribute code, please fork the repository and submit a pull request. I appreciate all contributions, from small bug fixes to major new features.

I look forward to collaborating with you!

## Acknowledgements

Obviously, the authors of any of the libraries & tools used in this project deserve credit, including but not limited to:
*   [libtorch](https://pytorch.org/cppdocs/)
*   [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
*   [Bazel](https://bazel.build/)

Additionally, kudos to Costa Huang who authored [CleanRL](https://github.com/vwxyzjn/cleanrl) which served as a baseline for comparing the results of this project.

## In Memory

This project is dedicated to my late Gran, who always supported my endeavours. I love you, Gran.
