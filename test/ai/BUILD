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
    ] + select({
        # cu130 libtorch DT_NEEDEDs the CUDA runtime; supply it so the test loads.
        "@platforms//os:linux": ["//third_party:cuda_runtime"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "gae-test",
    srcs = ["gae-test.cc"],
    copts = googletest_copts,
    deps = [
        "//src:ai",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ] + select({
        # cu130 libtorch DT_NEEDEDs the CUDA runtime; supply it so the test loads.
        "@platforms//os:linux": ["//third_party:cuda_runtime"],
        "//conditions:default": [],
    }),
)
