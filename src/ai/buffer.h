#include <torch/torch.h>
#include <vector>

namespace ai::buffer {

class Buffer {
public:
  Buffer(int capacity, std::vector<size_t> observation_shape, int action_size);

  void add(std::vector<unsigned char> state, int action, float reward) {
    states_[indices_] =
        torch::from_blob(state.data(), observation_shape_, torch::kByte);
    actions_[indices_] = torch::tensor(action);
    rewards_[indices_] = torch::tensor(reward);
    indices_ = (indices_ + 1) % capacity_;
  }

private:
  int capacity_;
  std::vector<int64_t> observation_shape_;
  torch::Tensor states_;
  torch::Tensor actions_;
  torch::Tensor rewards_;
  int indices_;
};

} // namespace ai::buffer