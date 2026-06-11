// Opt 1 microbenchmark -- reading the per-env selected actions on the host.
//
// rollout.cc step() used to call action_result_.actions[i].item<int64_t>() for
// every environment, i.e. N synchronising device->host scalar copies per step.
// The new path copies the whole actions tensor to the host once
// (actions_cpu_ = actions.to(kCPU)) and indexes that, i.e. a single sync.
#include <torch/torch.h>

#include <cstdio>

#include "bench/bench_util.h"

int main(int argc, char **argv) {
  auto device = bench::pick_device();
  int n = bench::arg_or(argc, argv, 1, 256);      // environments
  int iters = bench::arg_or(argc, argv, 2, 500);  // simulated rollout steps
  std::printf("opt1 actions->host   device=%s  envs=%d  iters=%d\n",
              device.is_cuda() ? "cuda" : "cpu", n, iters);

  // Fixed random action vector on the device (values don't affect timing).
  auto actions = torch::randint(
      0, 6, {n}, torch::TensorOptions(torch::kLong).device(device));

  // Correctness: both strategies must read identical action indices.
  int64_t old_sum = 0;
  for (int i = 0; i < n; ++i) old_sum += actions[i].item<int64_t>();
  int64_t new_sum = 0;
  {
    auto cpu = actions.to(torch::kCPU);
    for (int i = 0; i < n; ++i) new_sum += cpu[i].item<int64_t>();
  }
  std::printf("  correctness %s (checksum %ld)\n",
              old_sum == new_sum ? "OK" : "FAIL", static_cast<long>(old_sum));
  if (old_sum != new_sum) return 1;

  volatile int64_t sink = 0;
  double old_s = bench::time_loop(5, iters, [&]() {
    int64_t s = 0;
    for (int i = 0; i < n; ++i) s += actions[i].item<int64_t>();
    sink = s;
  });
  double new_s = bench::time_loop(5, iters, [&]() {
    int64_t s = 0;
    auto cpu = actions.to(torch::kCPU);
    for (int i = 0; i < n; ++i) s += cpu[i].item<int64_t>();
    sink = s;
  });
  (void)sink;
  bench::report("read N actions / step", old_s, new_s);
  return 0;
}
