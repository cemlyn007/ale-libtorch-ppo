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
        "//src:ai",
        "//src:stop_signal",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@spdlog",
        "@tensorboard_logger",
        "@yaml-cpp//:yaml-cpp",
    ],
)
