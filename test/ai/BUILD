load("@rules_cc//cc:defs.bzl", "cc_test")

googletest_copts = [
    "-Iexternal/gtest/googletest/include",
    "-Iexternal/gtest/googletest",
]

cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    copts = googletest_copts,
    data = [
        "//:roms",
    ],
    deps = [
        "//src/ai/environment",
        "//src/ai/environment:max_and_skip",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "state_io_test",
    srcs = ["state_io_test.cc"],
    # rollout.h pulls in torch headers flagged by GCC under -Werror.
    copts = googletest_copts + ["-Wno-error=interference-size"],
    data = [
        "//:roms",
    ],
    deps = [
        "//src/ai:rollout",
        "//src/ai/environment",
        "//src/ai/environment:episode_life",
        "//src/ai/environment:fire_reset",
        "//src/ai/environment:max_and_skip",
        "//src/ai/environment:noop_reset",
        "//src/ai/environment:resize",
        "//src/ai/environment:truncate_on_episode_return",
        "@arcade_learning_environment//:ale",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)

cc_test(
    name = "vision-test",
    srcs = ["vision-test.cc"],
    copts = googletest_copts,
    deps = [
        "//src:ai",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)

cc_test(
    name = "ppo-train-test",
    srcs = ["ppo-train-test.cc"],
    # train.h pulls in c10/cuda/CUDAEvent.h; see //src/ai/ppo:train.
    copts = googletest_copts + ["-Wno-error=interference-size"],
    deps = [
        "//src/ai/ppo",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)

cc_test(
    name = "gae-test",
    srcs = ["gae-test.cc"],
    copts = googletest_copts,
    deps = [
        "//src:ai",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)

cc_test(
    name = "losses-test",
    srcs = ["losses-test.cc"],
    copts = googletest_copts,
    deps = [
        "//src/ai/ppo:losses",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)

cc_test(
    name = "checkpoint-test",
    srcs = ["checkpoint-test.cc"],
    copts = googletest_copts,
    deps = [
        "//src/ai:checkpoint",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)

cc_test(
    name = "rollout-test",
    srcs = ["rollout-test.cc"],
    copts = googletest_copts,
    data = [
        "//:roms",
    ],
    deps = [
        "//src/ai:rollout",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)
