#include <torch/torch.h>
#include <vector>

namespace ai::vision {

torch::Tensor resize_grayscale_image(const torch::Tensor &);

// Note: That when using the GPU, the dtype must be a float.
torch::Tensor
resize_frame_stacked_grayscale_images(const torch::Tensor &images);
torch::Tensor resize_frame_stacked_rgb_images(const torch::Tensor &images);
torch::Tensor
rgb_to_grayscale_frame_stacked_images(const torch::Tensor &images);

std::vector<unsigned char>
resize_grayscale_image(const std::vector<unsigned char> &image, int width,
                       int height, int new_width, int new_height);

} // namespace ai::vision