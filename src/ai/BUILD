package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rollout",
    srcs = ["rollout.cc"],
    hdrs = ["rollout.h"],
    strip_include_prefix = "/src",
    deps = [
        ":buffer",
        ":gae",
        ":queue",
        "//src/ai/environment",
        "//src/ai/environment:episode_life",
        "//src/ai/environment:episode_observation_recorder",
        "//src/ai/environment:episode_recorder",
        "//src/ai/environment:fire_reset",
        "//src/ai/environment:max_and_skip",
        "//src/ai/environment:noop_reset",
        "//src/ai/environment:resize",
        "//src/ai/environment:truncate_on_episode_return",
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
    strip_include_prefix = "/src",
)

cc_library(
    name = "vision",
    srcs = ["vision.cc"],
    hdrs = ["vision.h"],
    copts = [
        "-DSTB_IMAGE_RESIZE_IMPLEMENTATION",
    ],
    strip_include_prefix = "/src",
    deps = [
        "@libtorch//:torch",
        "@stb//:stb_image_resize2",
    ],
)

cc_library(
    name = "queue",
    hdrs = ["queue.h"],
    strip_include_prefix = "/src",
)
