#pragma once
#include <cstddef>
#include <filesystem>
#include <string>

namespace training {

// The hyperparameters and run options for one PPO training run. Plain data:
// load_config() binds it from YAML and a tuner can mutate fields per arm.
struct Config {
  size_t total_environments;
  size_t hidden_size;
  size_t horizon;
  size_t max_steps;
  size_t frame_stack;
  double learning_rate;
  float clip_param;
  float value_loss_coef;
  float entropy_coef;
  long num_epochs;
  long mini_batch_size;
  long num_mini_batches;
  bool shuffle_mini_batches;
  float gae_discount;
  float gae_lambda;
  float max_gradient_norm;
  size_t num_rollouts;
  size_t num_workers;
  size_t worker_batch_size;
  // Contiguous env groups stepped as a software pipeline: while one group is
  // in the workers, inference and bookkeeping run for the others. 1 = the
  // classic fully-synchronous loop.
  size_t pipeline_groups;
  size_t frame_skip;
  // Some games like breakout have a maximum return
  // which should be used to reset the environment.
  float max_return;
  // It is faster to record using the observation.
  // However the observation may be in grayscale.
  bool record_observation;
  bool record_video;
  bool cuda_graph;
  // Double-buffered async PPO: run each update on a learner thread (side CUDA
  // stream) while the next rollout is collected with a snapshot of the policy.
  // Data is at most one policy version stale; the stored behaviour logits keep
  // the PPO ratio correct. false = the classic synchronous loop.
  bool async_update;
  bool deterministic;
  // Write latest.pt every `checkpoint_interval` rollouts (0 disables all
  // checkpointing) and best.pt whenever mean episode return improves, into a
  // run directory keyed by the same start_time stamp as the tfevents file.
  size_t checkpoint_interval;
  // Path to a checkpoint .pt to restore network + optimizer + step from; empty
  // starts fresh.
  std::string resume_from;
};

// The single place where YAML keys bind to Config members. Each field keeps its
// real type, so the loader and any logger stay strongly typed. action_size is
// excluded: it is a property of the ROM, not loaded from YAML.
template <typename Self, typename Visitor>
void for_each_field(Self &config, Visitor &&visit) {
  visit("total_environments", config.total_environments);
  visit("hidden_size", config.hidden_size);
  visit("horizon", config.horizon);
  visit("max_steps", config.max_steps);
  visit("frame_stack", config.frame_stack);
  visit("learning_rate", config.learning_rate);
  visit("clip_param", config.clip_param);
  visit("value_loss_coef", config.value_loss_coef);
  visit("entropy_coef", config.entropy_coef);
  visit("num_epochs", config.num_epochs);
  visit("mini_batch_size", config.mini_batch_size);
  visit("num_mini_batches", config.num_mini_batches);
  visit("shuffle_mini_batches", config.shuffle_mini_batches);
  visit("gae_discount", config.gae_discount);
  visit("gae_lambda", config.gae_lambda);
  visit("max_gradient_norm", config.max_gradient_norm);
  visit("num_rollouts", config.num_rollouts);
  visit("num_workers", config.num_workers);
  visit("worker_batch_size", config.worker_batch_size);
  visit("pipeline_groups", config.pipeline_groups);
  visit("frame_skip", config.frame_skip);
  visit("max_return", config.max_return);
  visit("record_observation", config.record_observation);
  visit("record_video", config.record_video);
  visit("cuda_graph", config.cuda_graph);
  visit("async_update", config.async_update);
  visit("deterministic", config.deterministic);
  visit("checkpoint_interval", config.checkpoint_interval);
  visit("resume_from", config.resume_from);
}

// Composes one Config from a training YAML and an environment YAML, the latter
// holding a game's truncation and frame settings so they live in one small file
// shared across training configs. The environment's keys win where the two
// overlap; a key missing from both files is a hard error.
Config compose_config(const std::filesystem::path &training_path,
                      const std::filesystem::path &environment_path);

// Rejects internally-inconsistent configs. Throws std::invalid_argument so a
// tuner can reject one bad arm without taking down the process.
void validate(const Config &config);

}  // namespace training
