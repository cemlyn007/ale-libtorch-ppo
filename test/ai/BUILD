googletest_copts = [
    "-Iexternal/gtest/googletest/include",
    "-Iexternal/gtest/googletest",
]

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
