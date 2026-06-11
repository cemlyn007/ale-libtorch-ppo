// Opt 3 microbenchmark -- writing every env's newest frame into the stack.
//
// rollout.cc used to torch::stack() N from_blob views into a fresh *pageable*
// host tensor, then index_put_ it into observations_[:, 0] -- so the per-env
// pinned_memory flag was a no-op and the bulk H2D ran from pageable memory. The
// new path keeps one genuinely page-locked staging tensor and issues a single
// asynchronous copy_ into the slice.
#include <torch/torch.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bench/bench_util.h"

int main(int argc, char **argv) {
  auto device = bench::pick_device();
  int n = bench::arg_or(argc, argv, 1, 256);          // environments
  int frame_stack = bench::arg_or(argc, argv, 2, 4);  // stacked frames
  int hw = bench::arg_or(argc, argv, 3, 84);          // frame height == width
  int iters = bench::arg_or(argc, argv, 4, 1000);     // simulated rollout steps
  std::printf(
      "opt3 frame upload    device=%s  envs=%d  stack=%d  hw=%dx%d  iters=%d\n",
      device.is_cuda() ? "cuda" : "cpu", n, frame_stack, hw, hw, iters);

  int frame_bytes = hw * hw;  // grayscale

  // OLD: per-env std::vector blobs wrapped with from_blob. The pinned_memory
  // flag here is a no-op -- from_blob never allocates, so the memory stays
  // pageable, exactly as in the original rollout.
  std::vector<std::vector<unsigned char>> bufs(n);
  std::vector<torch::Tensor> blobs(n);
  for (int i = 0; i < n; ++i) {
    bufs[i].resize(frame_bytes);
    for (int j = 0; j < frame_bytes; ++j)
      bufs[i][j] = static_cast<unsigned char>((i + j) & 0xff);
    blobs[i] = torch::from_blob(
        bufs[i].data(), {hw, hw},
        torch::TensorOptions(torch::kByte).pinned_memory(true));
  }

  // NEW: single page-locked staging buffer, filled with identical data.
  auto staging_opt = torch::TensorOptions(torch::kByte);
  if (device.is_cuda()) staging_opt = staging_opt.pinned_memory(true);
  auto staging = torch::empty({n, hw, hw}, staging_opt);
  auto *staging_ptr = staging.data_ptr<uint8_t>();
  for (int i = 0; i < n; ++i)
    std::memcpy(staging_ptr + static_cast<size_t>(i) * frame_bytes,
                bufs[i].data(), frame_bytes);

  auto byte_dev = torch::TensorOptions(torch::kByte).device(device);
  auto obs_old = torch::zeros({n, frame_stack, hw, hw}, byte_dev);
  auto obs_new = torch::zeros({n, frame_stack, hw, hw}, byte_dev);

  auto do_old = [&]() {
    obs_old.index_put_({torch::indexing::Slice(), 0}, torch::stack(blobs, 0));
  };
  auto do_new = [&]() {
    obs_new.select(1, 0).copy_(staging, /*non_blocking=*/true);
  };

  do_old();
  do_new();
  bench::sync();
  bool ok = obs_old.select(1, 0).equal(obs_new.select(1, 0));
  std::printf("  correctness %s\n", ok ? "OK" : "FAIL");
  if (!ok) return 1;

  double old_s = bench::time_loop(5, iters, do_old);
  double new_s = bench::time_loop(5, iters, do_new);
  bench::report("upload newest frame", old_s, new_s);
  return 0;
}
