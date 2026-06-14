#include "training/logging.h"

#include <torch/torch.h>

#include "ai/ppo/train.h"
#include "ai/rollout.h"
#include "ai/tensor_util.h"
#include "tensorboard_logger.h"

namespace training {

void log_rollout(TensorBoardLogger &logger, const ai::rollout::Log &log,
                 const ai::ppo::train::Metrics &metrics, double learning_rate,
                 bool histograms) {
  using ai::tensor_util::gather;
  using ai::tensor_util::mean;
  using ai::tensor_util::to_vector;
  const auto step = log.steps;
  const auto &masks = metrics.masks;
  auto scalar = [&](const char *tag, double v) {
    logger.add_scalar(tag, step, v);
  };
  auto hist = [&](const char *tag, const auto &v) {
    if (histograms) logger.add_histogram(tag, step, v);
  };
  auto g = [&](const torch::Tensor &t) { return gather(t, masks); };
  // Gather once on the host, then log both the mean and the distribution from
  // the same vector -- avoids a second masked_select + device sync per tensor.
  auto scalar_and_hist = [&](const char *mean_tag, const char *hist_tag,
                             const torch::Tensor &t) {
    auto v = g(t);
    scalar(mean_tag, mean(v));
    hist(hist_tag, v);
  };

  if (!log.episode_returns.empty()) {
    scalar("mean_episode_return", mean(log.episode_returns));
    scalar("mean_episode_length", mean(log.episode_lengths));
    hist("episode_returns", log.episode_returns);
    hist("episode_lengths", log.episode_lengths);
    if (!log.game_returns.empty()) {
      scalar("mean_game_return", mean(log.game_returns));
      scalar("mean_game_length", mean(log.game_lengths));
      hist("game_returns", log.game_returns);
      hist("game_lengths", log.game_lengths);
    }
  }

  scalar("mean_clipped_gradient",
         metrics.clipped_gradients.mean().item<float>());
  scalar("mean_loss", metrics.loss.mean().item<float>());
  scalar_and_hist("mean_clipped_loss", "clipped_losses",
                  metrics.clipped_losses);
  scalar_and_hist("mean_value_loss", "value_losses", metrics.value_losses);
  scalar_and_hist("mean_entropy", "entropies", metrics.entropies);
  scalar_and_hist("mean_ratio", "ratios", metrics.ratio);

  // Histogram-only series: each is a full masked gather to the host, so skip
  // the gather entirely (not just the write) when histograms are off.
  if (histograms) {
    if (metrics.clipped_gradients.numel() > 1)
      hist("clipped_gradients", to_vector(metrics.clipped_gradients));
    hist("losses", g(metrics.total_losses));
    hist("advantages", g(metrics.advantages));
    hist("returns", g(metrics.returns));
  }

  scalar("learning_rate", learning_rate);
}

}  // namespace training
