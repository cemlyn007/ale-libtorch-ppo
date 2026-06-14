load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ppo",
    strip_include_prefix = "/src",
    deps = [
        ":losses",
        ":train",
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "losses",
    srcs = ["losses.cc"],
    hdrs = ["losses.h"],
    strip_include_prefix = "/src",
    deps = [
        "@libtorch//:torch",
    ],
)

# Abstract training strategy the Loop drives; concrete implementations live with
# the application network (see //src/training:trainer).
cc_library(
    name = "trainer",
    hdrs = ["trainer.h"],
    strip_include_prefix = "/src",
    deps = [
        "//src/ai:buffer",
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "async_updater",
    srcs = ["async_updater.cc"],
    hdrs = ["async_updater.h"],
    # libtorch's c10/cuda/CUDAEvent.h uses
    # std::hardware_destructive_interference_size, flagged by GCC under -Werror.
    copts = ["-Wno-error=interference-size"],
    strip_include_prefix = "/src",
    deps = [
        ":train",
        "@libtorch//:torch",
    ] + select({
        "@platforms//os:linux": ["@cuda//:cuda_headers"],
        "//conditions:default": [],
    }),
)

# The generic PPO driver: collect/update/anneal/async pipeline. Agnostic to the
# concrete network and config — it only speaks to Trainer, Rollout and Metrics.
cc_library(
    name = "loop",
    srcs = ["loop.cc"],
    hdrs = ["loop.h"],
    copts = ["-Wno-error=interference-size"],
    strip_include_prefix = "/src",
    deps = [
        ":async_updater",
        ":train",
        ":trainer",
        "//src/ai:rollout",
        "//src/ai:tensor_util",
        "@libtorch//:torch",
    ] + select({
        "@platforms//os:linux": ["@cuda//:cuda_headers"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "train",
    srcs = ["train.cc"],
    hdrs = ["train.h"],
    # libtorch 2.11's c10/cuda/CUDAEvent.h uses
    # std::hardware_destructive_interference_size, which GCC flags under -Werror.
    copts = ["-Wno-error=interference-size"],
    strip_include_prefix = "/src",
    deps = [
        ":losses",
        "@libtorch//:torch",
    ] + select({
        "@platforms//os:linux": ["@cuda//:cuda_headers"],
        "//conditions:default": [],
    }),
)
