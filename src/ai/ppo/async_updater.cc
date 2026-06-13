#include "ai/ppo/async_updater.h"

#include <utility>

#include "ai/ppo/train.h"

namespace ai::ppo {

AsyncUpdater::AsyncUpdater(const torch::Device &device) {
#ifdef __linux__
  if (device.is_cuda()) stream_ = at::cuda::getStreamFromPool();
#else
  (void)device;
#endif
  thread_ = std::thread([this] { loop(); });
}

AsyncUpdater::~AsyncUpdater() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  condition_variable_.notify_one();
  thread_.join();
}

std::future<void> AsyncUpdater::submit(std::function<void()> job) {
  std::packaged_task<void()> task([this, job = std::move(job)] {
#ifdef __linux__
    if (stream_.has_value()) {
      auto main_stream = at::cuda::getDefaultCUDAStream();
      ai::ppo::train::stream_sync(main_stream, stream_.value());
      job();
      stream_->synchronize();
      return;
    }
#endif
    job();
  });
  auto future = task.get_future();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.push(std::move(task));
  }
  condition_variable_.notify_one();
  return future;
}

void AsyncUpdater::loop() {
#ifdef __linux__
  // Thread-local, so set once: every update this thread runs stays off the
  // default stream and can overlap the rollout's inference kernels.
  if (stream_.has_value()) at::cuda::setCurrentCUDAStream(stream_.value());
#endif
  while (true) {
    std::packaged_task<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_variable_.wait(lock,
                               [this] { return stop_ || !jobs_.empty(); });
      if (jobs_.empty()) return;  // stop_ set and queue drained
      task = std::move(jobs_.front());
      jobs_.pop();
    }
    task();  // exceptions land in the job's future
  }
}

}  // namespace ai::ppo
