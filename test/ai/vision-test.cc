#include "ai/vision.h"
#include "gtest/gtest.h"
#include <torch/torch.h>

TEST(VisionTest, ResizeGrayscaleImage) {
  torch::Tensor input = torch::ones({210, 160}, torch::kFloat32);
  torch::Tensor output = ai::vision::resize_grayscale_image(input);
  EXPECT_EQ(output.sizes(), std::vector<int64_t>({84, 84}));
  EXPECT_TRUE((output == 1).all().item<bool>());
}

TEST(VisionTest, ResizeGrayscaleImages) {
  torch::Tensor input = torch::empty({1, 4, 210, 160}, torch::kFloat32);
  for (uint8_t i = 0; i < 4; ++i) {
    EXPECT_EQ(input.index({0, i}).sizes(), std::vector<int64_t>({210, 160}));
    input.index_put_({0, i}, i);
    const auto &input_data = input.index({0, i});
    EXPECT_EQ(input_data.sizes(), std::vector<int64_t>({210, 160}));
    EXPECT_EQ(input_data.sum().item<int64_t>(), 210 * 160 * i);
  }
  torch::Tensor output =
      ai::vision::resize_frame_stacked_grayscale_images(input);
  EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 4, 84, 84}));
  EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 4, 84, 84}));
  EXPECT_EQ(output.dtype(), torch::kFloat32);
  for (int i = 0; i < 4; ++i) {
    auto output_data = output.index({0, i});
    EXPECT_EQ(output_data.sizes(), std::vector<int64_t>({84, 84}));
    EXPECT_TRUE((output_data == i).all().item<bool>());
  }
}