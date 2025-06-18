#include "ai/vision.h"
#include "gtest/gtest.h"
#include <torch/torch.h>

TEST(VisionTest, ResizeGrayscaleImages) {
  torch::Tensor input = torch::rand({1, 3, 210, 160});
  torch::Tensor output =
      ai::vision::resize_frame_stacked_grayscale_images(input);
  EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 84, 84}));
}