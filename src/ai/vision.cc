#include "ai/vision.h"
#include "stb_image_resize2.h"
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
}

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
}

// Expects a 5D tensor with shape
// [batch_size, frame_stack, channels, height, width].
// Returns a 5D tensor with shape
// [batch_size, frame_stack, channels, 84, 84].
torch::Tensor resize_frame_stacked_rgb_images(const torch::Tensor &images) {
  assert(images.dim() == 5);
  const auto batch_size = images.size(0);
  const auto frame_stack = images.size(1);
  assert(images.size(2) == 3);
  assert(images.size(3) == 210);
  assert(images.size(4) == 160);
  const auto input = images.flatten(0, 1);
  const auto out = torch::nn::functional::interpolate(input, options);
  return out.unflatten(0, {batch_size, frame_stack});
}

// https://github.com/Farama-Foundation/Gymnasium/blob/ad23dfbbe29f83107404f9f6a56131f6b498d0d7/gymnasium/wrappers/transform_observation.py#L320
const auto GRAYSCALE_WEIGHTS = torch::tensor({0.2125, 0.7154, 0.0721});

static std::unordered_map<torch::DeviceIndex, torch::Tensor>
    grayscale_weights_cache;

const torch::Tensor &get_grayscale_weights(torch::Device device) {
  auto device_idx = device.index();
  auto it = grayscale_weights_cache.find(device_idx);
  if (it == grayscale_weights_cache.end()) {
    auto device_weights = GRAYSCALE_WEIGHTS.to(device);
    grayscale_weights_cache[device_idx] = device_weights;
    return grayscale_weights_cache[device_idx];
  }
  return it->second;
}

// Expects a 5D tensor with shape
// [batch_size, frame_stack, channels, height, width].
// Returns a 4D tensor with shape
// [batch_size, frame_stack, height, width].
torch::Tensor
rgb_to_grayscale_frame_stacked_images(const torch::Tensor &images) {
  assert(images.dim() == 5);
  assert(images.size(2) == 3);
  assert(images.size(3) == 84);
  assert(images.size(4) == 84);
  const auto weights = get_grayscale_weights(images.device());
  const auto inputs = images.permute({0, 1, 3, 4, 2});
  // Or alternatively:
  // const auto grayscale_images = torch::sum(inputs * weights, -1);
  const auto grayscale_images = torch::matmul(inputs, weights);
  assert(grayscale_images.dim() == 4);
  return grayscale_images;
}

std::vector<unsigned char>
resize_grayscale_image(const std::vector<unsigned char> &image, int width,
                       int height, int new_width, int new_height) {
  assert(image.size() == static_cast<size_t>(width * height));
  std::vector<unsigned char> resized_image(new_width * new_height);
  stbir_resize_uint8_linear(image.data(), width, height, 0,
                            resized_image.data(), new_width, new_height, 0,
                            STBIR_1CHANNEL);
  return resized_image;
}

} // namespace ai::vision
