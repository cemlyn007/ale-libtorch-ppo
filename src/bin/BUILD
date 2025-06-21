load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ale_ppo",
    srcs = ["ale_ppo.cc"],
    deps = [
        "//src:ai",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@tensorboard_logger",
    ],
)

cc_binary(
    name = "ale_ppo_threaded",
    srcs = ["ale_ppo_threaded.cc"],
    deps = [
        "//src:ai",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@tensorboard_logger",
    ],
)
