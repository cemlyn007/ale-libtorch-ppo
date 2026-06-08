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
        "@googletest//:gtest_main",
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
    name = "checkpoint-test",
    srcs = ["checkpoint-test.cc"],
    copts = googletest_copts,
    deps = [
        "//src:checkpoint",
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
