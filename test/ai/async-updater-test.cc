#include "ai/ppo/async_updater.h"

#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <future>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

namespace {

using ai::ppo::AsyncUpdater;

// Every test drives the CPU path: with a CPU device the updater keeps no CUDA
// stream, so jobs run straight on its worker thread with no stream sync. The
// CUDA side-stream branch needs a GPU and is left to end-to-end runs.
torch::Device cpu() { return torch::Device(torch::kCPU); }

// A submitted job runs, and its future becomes ready once it has.
TEST(AsyncUpdater, RunsSubmittedJob) {
  AsyncUpdater updater(cpu());
  std::atomic<bool> ran{false};
  auto done = updater.submit([&ran] { ran = true; });
  done.get();
  EXPECT_TRUE(ran.load());
}

// The single worker drains jobs in submission order.
TEST(AsyncUpdater, PreservesSubmissionOrder) {
  AsyncUpdater updater(cpu());
  // Touched only by the worker, one job at a time; read after every future is
  // joined, so the accesses are ordered and race-free.
  std::vector<int> order;
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 50; ++i)
    futures.push_back(updater.submit([&order, i] { order.push_back(i); }));
  for (auto &f : futures) f.get();

  std::vector<int> expected(50);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(order, expected);
}

// A throwing job lands its exception in the future rather than crashing.
TEST(AsyncUpdater, PropagatesExceptionToFuture) {
  AsyncUpdater updater(cpu());
  auto done = updater.submit([] { throw std::runtime_error("boom"); });
  EXPECT_THROW(done.get(), std::runtime_error);
}

// submit() hands the work to another thread: the caller keeps running while the
// job is still blocked, and the future only completes once the job does.
TEST(AsyncUpdater, RunsJobOnAnotherThread) {
  AsyncUpdater updater(cpu());
  std::promise<void> release;
  std::future<void> released = release.get_future();
  std::atomic<bool> ran{false};
  auto done = updater.submit([&ran, &released] {
    released.wait();
    ran = true;
  });
  // The job cannot finish until we release it, so the future stays pending and
  // the side effect is unobserved here -- this is what overlap with the rollout
  // relies on.
  EXPECT_EQ(done.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_FALSE(ran.load());
  release.set_value();
  done.get();
  EXPECT_TRUE(ran.load());
}

// Destruction drains every queued job before the worker is joined, so work that
// was submitted and never waited on still runs.
TEST(AsyncUpdater, DrainsQueuedJobsOnDestroy) {
  std::atomic<int> count{0};
  {
    AsyncUpdater updater(cpu());
    for (int i = 0; i < 100; ++i) updater.submit([&count] { ++count; });
    // Futures dropped on purpose; the destructor must still run all 100.
  }
  EXPECT_EQ(count.load(), 100);
}

}  // namespace
