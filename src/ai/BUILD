package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rollout",
    srcs = ["rollout.cc"],
    hdrs = ["rollout.h"],
    strip_include_prefix = "/src",
    deps = [
        ":buffer",
        ":gae",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    strip_include_prefix = "/src",
    deps = [
        ":gae",
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "ppo",
    srcs = ["ppo.cc"],
    hdrs = ["ppo.h"],
    strip_include_prefix = "/src",
    deps = [
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "gae",
    srcs = ["gae.cc"],
    hdrs = ["gae.h"],
    strip_include_prefix = "/src",
    deps = [
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "vision",
    srcs = ["vision.cc"],
    hdrs = ["vision.h"],
    strip_include_prefix = "/src",
    deps = [
        "@libtorch//:torch",
    ],
)
