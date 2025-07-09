package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rollout",
    srcs = ["rollout.cc"],
    hdrs = ["rollout.h"],
    strip_include_prefix = "/src",
    deps = [
        ":buffer",
        ":environment",
        ":episode_life",
        ":episode_recorder",
        ":gae",
        ":queue",
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
    name = "environment",
    srcs = [
        "environment.cc",
    ],
    hdrs = [
        "environment.h",
    ],
    strip_include_prefix = "/src",
    deps = [
        "@arcade_learning_environment//:ale",
    ],
)

cc_library(
    name = "episode_life",
    srcs = [
        "episode_life.cc",
    ],
    hdrs = [
        "episode_life.h",
    ],
    strip_include_prefix = "/src",
    deps = [
        ":environment",
        "@arcade_learning_environment//:ale",
    ],
)

cc_library(
    name = "episode_recorder",
    srcs = [
        "episode_recorder.cc",
    ],
    hdrs = [
        "episode_recorder.h",
    ],
    strip_include_prefix = "/src",
    deps = [
        ":environment",
        ":video_recorder",
        "@arcade_learning_environment//:ale",
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
    name = "video_recorder",
    srcs = ["video_recorder.cc"],
    hdrs = ["video_recorder.h"],
    copts = [
        "-DSTB_IMAGE_IMPLEMENTATION",
        "-DSTB_IMAGE_WRITE_IMPLEMENTATION",
    ],
    strip_include_prefix = "/src",
    deps = [
        "@stb//:stb_image",
        "@stb//:stb_image_write",
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

cc_library(
    name = "queue",
    hdrs = ["queue.h"],
    strip_include_prefix = "/src",
)
