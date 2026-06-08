#pragma once
#include <cstddef>
#include <filesystem>
#include <torch/torch.h>

namespace checkpoint {

// Everything needed to resume a run: model weights, optimizer moments, the next
// rollout to run (so the LR schedule continues), and the best return seen so the
// best.pt criterion survives a resume.
struct Checkpoint {
  size_t next_rollout_index;
  double best_return;
  // Absolute global env step at save time, so a resumed run's TensorBoard
  // curves continue from here instead of restarting at 0.
  size_t global_step;
};

// Takes the model by its torch::nn::Module base rather than the concrete network
// type so this stays decoupled from the architecture — callers pass `*network`.
inline void save(const std::filesystem::path &path,
                 const torch::nn::Module &network,
                 const torch::optim::Adam &optimizer, const Checkpoint &state) {
  torch::serialize::OutputArchive archive;
  torch::serialize::OutputArchive model_archive;
  network.save(model_archive);
  archive.write("model", model_archive);
  torch::serialize::OutputArchive optimizer_archive;
  optimizer.save(optimizer_archive);
  archive.write("optimizer", optimizer_archive);
  archive.write("next_rollout_index",
                c10::IValue(static_cast<int64_t>(state.next_rollout_index)));
  archive.write("best_return", c10::IValue(state.best_return));
  archive.write("global_step",
                c10::IValue(static_cast<int64_t>(state.global_step)));
  // Write to a sibling temp file then rename: rename is atomic on a single
  // filesystem, so a crash mid-write can never truncate an existing checkpoint.
  auto tmp = path;
  tmp += ".tmp";
  archive.save_to(tmp.string());
  std::filesystem::rename(tmp, path);
}

inline Checkpoint load(const std::filesystem::path &path,
                       torch::nn::Module &network,
                       torch::optim::Adam &optimizer,
                       const torch::Device &device) {
  torch::serialize::InputArchive archive;
  // Remap storages onto the current device so a CPU-saved checkpoint resumes on
  // GPU and vice versa. Module::load copies into the existing parameters in
  // place, keeping the tensors the optimizer already references valid.
  archive.load_from(path.string(), device);
  torch::serialize::InputArchive model_archive;
  archive.read("model", model_archive);
  network.load(model_archive);
  torch::serialize::InputArchive optimizer_archive;
  archive.read("optimizer", optimizer_archive);
  optimizer.load(optimizer_archive);
  c10::IValue next_rollout_index, best_return, global_step;
  archive.read("next_rollout_index", next_rollout_index);
  archive.read("best_return", best_return);
  archive.read("global_step", global_step);
  return {static_cast<size_t>(next_rollout_index.toInt()),
          best_return.toDouble(), static_cast<size_t>(global_step.toInt())};
}

} // namespace checkpoint
