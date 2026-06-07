load("@rules_cc//cc:defs.bzl", "cc_library")

# cuSPARSELt (CUDA 13 redist). libtorch_cuda.so DT_NEEDEDs libcusparseLt.so.0.
cc_library(
    name = "cusparselt",
    srcs = glob([
        "lib/*.so",
        "lib/*.so.*",
    ]),
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)
