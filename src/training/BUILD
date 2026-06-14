load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# The concrete experiment: this run's hyperparameter schema, the Atari network,
# the trainer strategies that bind them, and the Session that wires everything
# onto the generic //src/ai/ppo:loop. Kept out of //src/ai so that library
# stays agnostic to the config type and network implementation.

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    strip_include_prefix = "/src",
    deps = ["@yaml-cpp//:yaml-cpp"],
)

cc_library(
    name = "network",
    hdrs = ["network.h"],
    strip_include_prefix = "/src",
    deps = ["@libtorch//:torch"],
)

# Bandit-based HPO controller (Successive Halving). Pure orchestration over the
# Config search space; the evaluator is injected, so it never links the training
# stack and stays unit-testable without a GPU.
cc_library(
    name = "bandit",
    srcs = ["bandit.cc"],
    hdrs = ["bandit.h"],
    strip_include_prefix = "/src",
    deps = [
        ":config",
        "@yaml-cpp//:yaml-cpp",
    ],
)

# TensorBoard logging of a rollout's metrics, shared by the trainer and the
# bandit sweep so both emit identical scalar series (the sweep skips histograms
# to keep per-arm logging cheap). Knows the TB tag schema, so it lives here
# rather than in the agnostic //src/ai layer.
cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    copts = [
        # protobuf headers (via tensorboard_logger) trip -Wdeprecated-declarations.
        "-Wno-error=deprecated-declarations",
        # libtorch's c10/cuda/CUDAEvent.h (via ai/ppo/train.h) uses
        # std::hardware_destructive_interference_size, flagged under -Werror.
        "-Wno-error=interference-size",
    ],
    strip_include_prefix = "/src",
    deps = [
        "//src/ai:rollout",
        "//src/ai:tensor_util",
        "//src/ai/ppo:train",
        "@libtorch//:torch",
        "@tensorboard_logger",
    ],
)

cc_library(
    name = "trainer",
    srcs = ["trainer.cc"],
    hdrs = ["trainer.h"],
    copts = ["-Wno-error=interference-size"],
    strip_include_prefix = "/src",
    deps = [
        ":config",
        ":network",
        "//src/ai:buffer",
        "//src/ai/ppo:losses",
        "//src/ai/ppo:train",
        "//src/ai/ppo:trainer",
        "@libtorch//:torch",
        "@spdlog",
    ] + select({
        "@platforms//os:linux": ["@cuda//:cuda_headers"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "session",
    srcs = ["session.cc"],
    hdrs = ["session.h"],
    copts = ["-Wno-error=interference-size"],
    strip_include_prefix = "/src",
    deps = [
        ":config",
        ":network",
        ":trainer",
        "//src/ai:rollout",
        "//src/ai:torch_runtime",
        "//src/ai/ppo:async_updater",
        "//src/ai/ppo:loop",
        "//src/ai/ppo:train",
        "//src/ai/ppo:trainer",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
    ] + select({
        "@platforms//os:linux": ["@cuda//:cuda_headers"],
        "//conditions:default": [],
    }),
)
