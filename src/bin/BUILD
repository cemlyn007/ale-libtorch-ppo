load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "train",
    srcs = ["train.cc"],
    # protobuf 33.4 headers (via tensorboard_logger) trip -Wdeprecated-declarations.
    copts = ["-Wno-error=deprecated-declarations"],
    deps = [
        "//src:ai",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@tensorboard_logger",
        "@yaml-cpp//:yaml-cpp",
    ],
)
