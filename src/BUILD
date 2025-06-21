load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ale_example",
    srcs = ["ale_example.cc"],
    deps = [
        "@arcade_learning_environment//:ale",
    ],
)

cc_binary(
    name = "ale_libtorch_example",
    srcs = ["ale_libtorch_example.cc"],
    deps = [
        ":ai",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@tensorboard_logger",
    ],
)

cc_binary(
    name = "libtorch_example",
    srcs = [
        "libtorch_example.cc",
    ],
    deps = [
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "ai",
    strip_include_prefix = "/src",
    deps = [
        "//src/ai:buffer",
        "//src/ai:gae",
        "//src/ai:rollout",
        "//src/ai:video_recorder",
        "//src/ai:vision",
        "//src/ai/ppo",
    ],
)
