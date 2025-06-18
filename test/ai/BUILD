cc_test(
    name = "vision-test",
    srcs = ["vision-test.cc"],
    copts = [
        "-Iexternal/gtest/googletest/include",
        "-Iexternal/gtest/googletest",
    ],
    deps = [
        "//src:ai",
        "@googletest//:gtest_main",
        "@libtorch//:torch",
    ],
)
