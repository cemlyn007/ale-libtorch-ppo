load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ale_ppo_threaded",
    srcs = ["ale_ppo_threaded.cc"],
    deps = [
        ":ai",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@tensorboard_logger",
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
