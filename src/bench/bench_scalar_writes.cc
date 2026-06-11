// Opt 2 microbenchmark -- uploading per-env reward/terminal/truncation state.
//
// rollout.cc used to write rewards_[i], is_terminated_[i], is_truncated_[i] one
// scalar at a time, i.e. ~3N synchronising host->device writes per step. The
// new path mutates persistent host mirrors and uploads each with a single bulk
// copy_ over a from_blob view, i.e. 3 copies regardless of env count.
#include <torch/torch.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bench/bench_util.h"

int main(int argc, char **argv) {
  auto device = bench::pick_device();
  int n = bench::arg_or(argc, argv, 1, 256);      // environments
  int iters = bench::arg_or(argc, argv, 2, 500);  // simulated rollout steps
  std::printf("opt2 scalar writes   device=%s  envs=%d  iters=%d\n",
              device.is_cuda() ? "cuda" : "cpu", n, iters);

  auto fopt = torch::TensorOptions(torch::kFloat32).device(device);
  auto bopt = torch::TensorOptions(torch::kBool).device(device);
  auto rewards_old = torch::zeros({n}, fopt);
  auto term_old = torch::zeros({n}, bopt);
  auto trunc_old = torch::zeros({n}, bopt);
  auto rewards_new = torch::zeros({n}, fopt);
  auto term_new = torch::zeros({n}, bopt);
  auto trunc_new = torch::zeros({n}, bopt);

  // Synthetic per-env step results held in host mirrors (as in the new
  // rollout).
  std::vector<float> r(n);
  std::vector<uint8_t> te(n), tr(n);
  for (int i = 0; i < n; ++i) {
    r[i] = 0.5f * static_cast<float>(i);
    te[i] = (i % 3 == 0);
    tr[i] = (i % 5 == 0);
  }
  auto byte = torch::TensorOptions(torch::kByte);

  auto do_old = [&]() {
    for (int i = 0; i < n; ++i) {
      rewards_old[i] = r[i];
      term_old[i] = static_cast<bool>(te[i]);
      trunc_old[i] = static_cast<bool>(tr[i]);
    }
  };
  auto do_new = [&]() {
    rewards_new.copy_(torch::from_blob(r.data(), {n}, torch::kFloat32));
    term_new.copy_(torch::from_blob(te.data(), {n}, byte));
    trunc_new.copy_(torch::from_blob(tr.data(), {n}, byte));
  };

  do_old();
  do_new();
  bench::sync();
  bool ok = rewards_old.equal(rewards_new) && term_old.equal(term_new) &&
            trunc_old.equal(trunc_new);
  std::printf("  correctness %s\n", ok ? "OK" : "FAIL");
  if (!ok) return 1;

  double old_s = bench::time_loop(5, iters, do_old);
  double new_s = bench::time_loop(5, iters, do_new);
  bench::report("upload reward/term/trunc", old_s, new_s);
  return 0;
}
