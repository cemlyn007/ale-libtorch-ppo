#include <condition_variable>
#include <mutex>
#include <queue>

namespace ai::queue {
template <typename T> class Queue {
private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable condition_variable_;

public:
  void push(const T &item) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(item);
    condition_variable_.notify_one();
  }

  void push(const std::vector<T> &items) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (const auto &item : items)
      queue_.push(item);
    condition_variable_.notify_all();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_variable_.wait(lock, [this]() { return !queue_.empty(); });
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  std::vector<T> pop(size_t n) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_variable_.wait(lock, [this, n]() { return queue_.size() >= n; });
    n = std::min(n, queue_.size());
    std::vector<T> items(n);
    for (size_t i = 0; i < n; ++i) {
      items[i] = std::move(queue_.front());
      queue_.pop();
    }
    return items;
  }
};
} // namespace ai::queue