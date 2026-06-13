load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "train",
    srcs = ["train.cc"],
    copts = [
        # protobuf 33.4 headers (via tensorboard_logger) trip -Wdeprecated-declarations.
        "-Wno-error=deprecated-declarations",
        # libtorch 2.11's c10/cuda/CUDAEvent.h uses
        # std::hardware_destructive_interference_size, which GCC flags under -Werror.
        "-Wno-error=interference-size",
    ],
    deps = [
        "//src:stop_signal",
        "//src/ai:checkpoint",
        "//src/ai:rollout",
        "//src/ai:tensor_util",
        "//src/ai:torch_runtime",
        "//src/ai/ppo:loop",
        "//src/ai/ppo:train",
        "//src/training:config",
        "//src/training:session",
        "@arcade_learning_environment//:ale",
        "@cli11//:cli11",
        "@libtorch//:torch",
        "@spdlog",
        "@tensorboard_logger",
    ],
)
