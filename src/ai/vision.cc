#include "ai/vision.h"
#include <array>
#include <torch/torch.h>

namespace ai::vision {

const auto options = torch::nn::functional::InterpolateFuncOptions()
                         .size(std::vector<int64_t>({84, 84}))
                         .mode(torch::kArea);

torch::Tensor resize_grayscale_image(const torch::Tensor &image) {
  assert(image.dim() == 2);
  assert(image.size(0) == 210);
  assert(image.size(1) == 160);
  const std::array<int64_t, 4> size = {1, 1, 210, 160};
  const auto &input = image.view(size);
  const auto out = torch::nn::functional::interpolate(input, options);
  return out.reshape({84, 84});
};

torch::Tensor
resize_frame_stacked_grayscale_images(const torch::Tensor &images) {
  assert(images.dim() == 4);
  const auto batch_size = images.size(0);
  const auto frame_stack = images.size(1);
  assert(images.size(2) == 210);
  assert(images.size(3) == 160);
  const auto input = images.flatten(0, 1).unsqueeze(1);
  const auto out = torch::nn::functional::interpolate(input, options);
  return out.squeeze(1).unflatten(0, {batch_size, frame_stack});
};

} // namespace ai::vision
