#pragma once

namespace ai::rollout {
struct Log;
}
namespace ai::ppo::train {
struct Metrics;
}
class TensorBoardLogger;

namespace training {

// Write one rollout's diagnostics to TensorBoard: episode/game return and
// length, the loss breakdown (clipped / value / entropy / ratio), gradient norm
// and learning rate. With `histograms` it also emits the full per-sample
// distributions -- several extra host gathers and far more data per rollout, so
// it is on for the main trainer but off for the bandit sweep, where every arm
// logs every rollout and throughput drives the score.
void log_rollout(TensorBoardLogger &logger, const ai::rollout::Log &log,
                 const ai::ppo::train::Metrics &metrics, double learning_rate,
                 bool histograms = true);

}  // namespace training
