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
    strip_include_prefix = "/src",
    deps = [
        ":losses",
        "@cuda//:cuda_headers",
        "@libtorch//:torch",
    ],
)
