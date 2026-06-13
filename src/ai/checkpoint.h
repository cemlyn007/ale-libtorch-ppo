#pragma once
#include <torch/torch.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace ai::checkpoint {

// Everything needed to resume a run: model weights, optimizer moments, the next
// rollout to run (so the LR schedule continues), and the best return seen so
// the best.pt criterion survives a resume.
struct Checkpoint {
  size_t next_rollout_index;
  double best_return;
  // Absolute global env step at save time, so a resumed run's TensorBoard
  // curves continue from here instead of restarting at 0.
  size_t global_step;
};

// Takes the model by its torch::nn::Module base rather than the concrete
// network type so this stays decoupled from the architecture — callers pass
// `*network`.
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

// Owns the per-run checkpoint policy so the training loop only reports rollout
// results: writes best.pt whenever the mean episode return improves and
// latest.pt every `interval` rollouts into `run_dir`. An interval of 0 disables
// checkpointing entirely.
class Checkpointer {
 public:
  // Called once per file written with the global step and a human-readable
  // description, so the caller can surface writes (e.g. to TensorBoard)
  // without this class depending on a logger.
  using Announce = std::function<void(size_t step, const std::string &)>;

  // initial_best_return seeds the best.pt criterion so it survives a resume:
  // pass the restored checkpoint's best_return, or leave it at -inf to start
  // fresh.
  Checkpointer(
      std::filesystem::path run_dir, size_t interval, Announce announce = {},
      double initial_best_return = -std::numeric_limits<double>::infinity())
      : run_dir_(std::move(run_dir)),
        interval_(interval),
        announce_(std::move(announce)),
        best_return_(initial_best_return) {}

  // Mean episode return of the best rollout so far.
  double best_return() const { return best_return_; }

  // rollout_return is empty when the rollout finished no episodes, in which
  // case best.pt cannot be judged and only the latest.pt cadence applies.
  void on_rollout_end(size_t rollout_index, size_t global_step,
                      std::optional<double> rollout_return,
                      const torch::nn::Module &network,
                      const torch::optim::Adam &optimizer) {
    if (interval_ == 0) return;
    const size_t next_rollout_index = rollout_index + 1;
    if (rollout_return.has_value() && *rollout_return > best_return_) {
      best_return_ = *rollout_return;
      save(run_dir_ / "best.pt", network, optimizer,
           {next_rollout_index, best_return_, global_step});
      announce(global_step,
               "best.pt return=" + std::to_string(best_return_) +
                   " rollout=" + std::to_string(next_rollout_index));
    }
    // Remember the newest rollout so flush_latest can persist it on exit even
    // when it did not land on an interval boundary.
    pending_next_rollout_index_ = next_rollout_index;
    pending_global_step_ = global_step;
    pending_latest_ = true;
    if (next_rollout_index % interval_ == 0)
      flush_latest(network, optimizer, "interval");
  }

  // Write latest.pt for the most recently completed rollout when it has not
  // already been written at an interval. A graceful stop breaks before the next
  // interval and a clean run can finish between intervals; either way this
  // keeps the newest weights rather than only the last interval multiple.
  // `reason` is surfaced in the announce text. No-op when checkpointing is
  // disabled or nothing is pending (so calling it after an interval save never
  // duplicates).
  void flush_latest(const torch::nn::Module &network,
                    const torch::optim::Adam &optimizer,
                    const std::string &reason) {
    if (interval_ == 0 || !pending_latest_) return;
    save(run_dir_ / "latest.pt", network, optimizer,
         {pending_next_rollout_index_, best_return_, pending_global_step_});
    announce(
        pending_global_step_,
        "latest.pt rollout=" + std::to_string(pending_next_rollout_index_) +
            " (" + reason + ")");
    pending_latest_ = false;
  }

 private:
  void announce(size_t step, const std::string &text) {
    if (announce_) announce_(step, text);
  }

  std::filesystem::path run_dir_;
  size_t interval_;
  Announce announce_;
  double best_return_;
  // The latest completed rollout, awaiting a latest.pt write. pending_latest_
  // is cleared once written so an interval save and a later flush never
  // collide.
  bool pending_latest_ = false;
  size_t pending_next_rollout_index_ = 0;
  size_t pending_global_step_ = 0;
};

}  // namespace ai::checkpoint
