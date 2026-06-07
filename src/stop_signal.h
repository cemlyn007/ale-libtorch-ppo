#pragma once
#include <atomic>
#include <csignal>
#include <initializer_list>

namespace stop_signal {

// Turns the given signals into a cooperative stop: the handler only flips a
// flag, so the main loop can break at a safe boundary and let RAII finalize
// (e.g. flush the ffmpeg mp4 trailer) instead of dying mid-write.
//
// The flag is static because a POSIX handler has no `this`; this type is
// therefore effectively a singleton — construct exactly one.
class StopSignal {
public:
  explicit StopSignal(std::initializer_list<int> signals) {
    struct sigaction sa{};
    sa.sa_handler = &StopSignal::handle;
    sigemptyset(&sa.sa_mask);
    // A second signal restores the default action and hard-kills, so we are
    // never stuck if finalizing hangs.
    sa.sa_flags = SA_RESETHAND;
    for (int signal : signals)
      sigaction(signal, &sa, nullptr);
    // A dying ffmpeg must not kill us with SIGPIPE before we finalize.
    std::signal(SIGPIPE, SIG_IGN);
  }

  bool requested() const { return flag_.load(std::memory_order_relaxed); }

private:
  static void handle(int) { flag_.store(true, std::memory_order_relaxed); }
  static inline std::atomic<bool> flag_{false};
};

} // namespace stop_signal
