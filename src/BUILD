load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "ale_example",
    srcs = ["ale_example.cc"],
    copts = [
        "-std=c++20",
    ],
    linkopts = ["-lz"],
    deps = [
        "@arcade_learning_environment//:ale",
        "@zlib",
    ],
)

cc_binary(
    name = "libtorch_example",
    srcs = [
        "libtorch_example.cc",
    ],
    deps = [
        "@libtorch//:torch",
    ],
)
