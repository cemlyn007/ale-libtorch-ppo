#pragma once
#include "ai/buffer.h"

namespace ai::ppo {

// Strategy for turning a fresh rollout into one optimization pass, abstracted
// away from the concrete network/optimizer so the training loop never names
// them. Concrete implementations (eager, CUDA-graph) live with the application
// network they drive; the loop only ever talks to this interface.
struct Trainer {
  virtual ~Trainer() = default;
  // Schedule the next learning rate. Honoured eagerly; a no-op once a CUDA
  // graph has baked the rate in at capture time.
  virtual void set_learning_rate(double lr) = 0;
  // The learning rate actually in effect, for honest logging.
  virtual double learning_rate() = 0;
  // Refresh inputs from the latest rollout and run one optimization pass,
  // writing per-update metrics into the shared Metrics buffer.
  virtual void update(ai::buffer::Batch &rollout) = 0;
};

}  // namespace ai::ppo
