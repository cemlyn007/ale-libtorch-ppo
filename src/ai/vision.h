#include <torch/torch.h>

namespace ai::vision {

// Note: That when using the GPU, the dtype must be a float.
torch::Tensor
resize_frame_stacked_grayscale_images(const torch::Tensor &images);

} // namespace ai::vision