#include "ai/vision.h"
#include "torch/torch.h"
#include <array>

namespace ai::vision {

torch::Tensor resize_grayscale_image(const torch::Tensor &image) {
  assert(image.dim() == 2);
  assert(image.size(0) == 210);
  assert(image.size(1) == 160);
  const auto options = torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>({84, 84}))
                           .mode(torch::kNearestExact)
                           .align_corners(false);
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
  const auto options = torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>({84, 84}))
                           .mode(torch::kNearestExact)
                           .align_corners(false);
  const std::array<int64_t, 4> size = {batch_size * frame_stack, 1, 210, 160};
  const auto input = images.view(size);
  const auto out = torch::nn::functional::interpolate(input, options);
  const std::array<int64_t, 4> shape = {batch_size, frame_stack, 84, 84};
  return out.reshape(shape);
};

} // namespace ai::vision
