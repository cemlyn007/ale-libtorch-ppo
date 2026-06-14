#pragma once
#include <torch/torch.h>

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#ifdef __linux__
#include <c10/cuda/CUDAStream.h>

#include <optional>
#endif

namespace ai::ppo {

// Runs jobs on a dedicated thread so the main thread can collect the next
// rollout concurrently. On CUDA the jobs run on a side stream, ordered after
// everything the main thread has enqueued; each job host-syncs that stream
// before fulfilling its future, so a joined future means its GPU work is done.
class AsyncUpdater {
 public:
  explicit AsyncUpdater(const torch::Device &device);
  ~AsyncUpdater();

  std::future<void> submit(std::function<void()> job);

 private:
  void loop();

#ifdef __linux__
  std::optional<at::cuda::CUDAStream> stream_;
#endif
  std::mutex mutex_;
  std::condition_variable condition_variable_;
  std::queue<std::packaged_task<void()>> jobs_;
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace ai::ppo
