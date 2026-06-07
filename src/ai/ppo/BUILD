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
        "@platforms//os:linux": ["//third_party:cuda_headers"],
        "//conditions:default": [],
    }),
)
