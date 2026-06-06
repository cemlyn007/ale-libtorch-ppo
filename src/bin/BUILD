load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "train",
    srcs = ["train.cc"],
    # protobuf 33.4 headers (pulled in via tensorboard_logger) trigger
    # -Wdeprecated-declarations from their own internal use of a deprecated
    # API; don't let that third-party warning fail our -Werror build.
    copts = ["-Wno-error=deprecated-declarations"],
    deps = [
        "//src:ai",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@tensorboard_logger",
        "@yaml-cpp//:yaml-cpp",
    ],
)
