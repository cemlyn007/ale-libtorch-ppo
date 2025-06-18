#include "torch/torch.h"
#include <array>

namespace ai::vision {

torch::Tensor
resize_frame_stacked_grayscale_images(const torch::Tensor &images) {
  assert(images.dim() == 4);
  assert(images.size(2) == 210);
  assert(images.size(3) == 160);
  const auto batch_size = images.size(0);
  const auto frame_stack = images.size(1);
  const auto options = torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>({84, 84}))
                           .mode(torch::kBilinear)
                           .align_corners(false);
  const std::array<int64_t, 4> size = {batch_size * frame_stack, 1, 210, 160};
  const auto input = images.view(size);
  const auto out = torch::nn::functional::interpolate(input, options);
  const std::array<int64_t, 4> shape = {batch_size, frame_stack, 84, 84};
  return out.reshape(shape);
};

} // namespace ai::vision
