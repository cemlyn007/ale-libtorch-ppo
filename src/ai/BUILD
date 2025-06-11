package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rollout",
    srcs = ["rollout.cc"],
    hdrs = ["rollout.h"],
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
    deps = [
        "@arcade_learning_environment//:ale",
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "ppo",
    srcs = ["ppo.cc"],
    hdrs = ["ppo.h"],
    deps = [
        "@libtorch//:torch",
    ],
)

cc_library(
    name = "gae",
    srcs = ["gae.cc"],
    hdrs = ["gae.h"],
    deps = [
        "@libtorch//:torch",
    ],
)
