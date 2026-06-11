#include <torch/cuda.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Shared helpers for the rollout CPU<->GPU bridge microbenchmarks. Each
// benchmark reproduces, on synthetic tensors, the exact operations of one
// optimization in src/ai/rollout.cc so the old and new paths can be timed (and
// checked for equivalence) in isolation.
namespace bench {

inline torch::Device pick_device() {
  return torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                     : torch::Device(torch::kCPU);
}

// Block until all queued device work is done -- required before stopping a
// timer, since copy_(non_blocking=true) and kernel launches are asynchronous.
inline void sync() {
  if (torch::cuda::is_available()) torch::cuda::synchronize();
}

using Clock = std::chrono::steady_clock;

// Mean wall-clock seconds per call of `fn`, device-synchronised around the
// timed loop so asynchronous GPU work is fully accounted for.
template <typename F>
double time_loop(int warmup, int iters, F &&fn) {
  for (int i = 0; i < warmup; ++i) fn();
  sync();
  auto start = Clock::now();
  for (int i = 0; i < iters; ++i) fn();
  sync();
  std::chrono::duration<double> elapsed = Clock::now() - start;
  return elapsed.count() / iters;
}

inline int arg_or(int argc, char **argv, int index, int fallback) {
  return argc > index ? std::atoi(argv[index]) : fallback;
}

inline void report(const std::string &name, double old_s, double new_s) {
  std::printf("  %-26s old %9.2f us   new %9.2f us   speedup %6.2fx\n",
              name.c_str(), old_s * 1e6, new_s * 1e6, old_s / new_s);
}

}  // namespace bench
