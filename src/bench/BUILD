load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

# libtorch 2.11's c10/cuda/CUDAEvent.h uses
# std::hardware_destructive_interference_size, which GCC flags under -Werror.
_TORCH_COPTS = ["-Wno-error=interference-size"]

cc_library(
    name = "bench_util",
    hdrs = ["bench_util.h"],
    copts = _TORCH_COPTS,
    strip_include_prefix = "/src",
    deps = ["@libtorch//:torch"],
)

cc_binary(
    name = "bench_actions",
    srcs = ["bench_actions.cc"],
    copts = _TORCH_COPTS,
    deps = [
        ":bench_util",
        "@libtorch//:torch",
    ],
)

cc_binary(
    name = "bench_scalar_writes",
    srcs = ["bench_scalar_writes.cc"],
    copts = _TORCH_COPTS,
    deps = [
        ":bench_util",
        "@libtorch//:torch",
    ],
)

cc_binary(
    name = "bench_staging",
    srcs = ["bench_staging.cc"],
    copts = _TORCH_COPTS,
    deps = [
        ":bench_util",
        "@libtorch//:torch",
    ],
)
