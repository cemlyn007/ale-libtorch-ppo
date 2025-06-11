load("@rules_cc//cc:defs.bzl", "cc_binary")

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
    deps = [
        "//src/ai:buffer",
        "//src/ai:ppo",
        "//src/ai:rollout",
    ],
)
