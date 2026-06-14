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
        "//src/ai:checkpoint",
        "//src/ai:rollout",
        "//src/ai:torch_runtime",
        "//src/ai/ppo:async_updater",
        "//src/ai/ppo:loop",
        "//src/ai/ppo:train",
        "//src/ai/ppo:trainer",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
        "@spdlog",
    ] + select({
        "@platforms//os:linux": ["@cuda//:cuda_headers"],
        "//conditions:default": [],
    }),
)
