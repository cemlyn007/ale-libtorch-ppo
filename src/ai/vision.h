#include <torch/torch.h>

namespace ai::vision {

torch::Tensor resize_grayscale_image(const torch::Tensor &);

// Note: That when using the GPU, the dtype must be a float.
torch::Tensor
resize_frame_stacked_grayscale_images(const torch::Tensor &images);
torch::Tensor resize_frame_stacked_rgb_images(const torch::Tensor &images);
torch::Tensor
rgb_to_grayscale_frame_stacked_images(const torch::Tensor &images);

} // namespace ai::vision