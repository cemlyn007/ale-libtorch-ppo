package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "torch",
    srcs = select({
        "@platforms//os:linux": glob(
            ["lib/*"],
            exclude = [
                "lib/libtorch_cpu.so",
                "lib/libc10.so",
                "lib/libnnapi_backend.so",
                "lib/libnvrtc-builtins.so",
                "lib/libtorch_python.so",
            ],
        ),
        "@platforms//os:macos": ["lib/libtorch.dylib"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    hdrs = glob([
        "include/torch/**",
        "include/ATen/**",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    deps = [
        ":c10",
        ":omp",
        ":torch_cpu",
    ],
    alwayslink = True,
)

cc_library(
    name = "torch_cpu",
    srcs = select({
        "@platforms//os:linux": ["lib/libtorch_cpu.so"],
        "@platforms//os:macos": ["lib/libtorch_cpu.dylib"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    hdrs = glob([
        "include/torch/**",
        "include/ATen/**",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    deps = [
        ":c10",
        ":omp",
    ],
    alwayslink = True,
)

cc_import(
    name = "c10",
    hdrs = glob(["include/c10/**"]),
    includes = ["include"],
    shared_library = select({
        "@platforms//os:linux": "lib/libc10.so",
        "@platforms//os:macos": "lib/libc10.dylib",
        "//conditions:default": "@platforms//:incompatible",
    }),
)

cc_import(
    name = "omp",
    shared_library = select({
        "@platforms//os:linux": "lib/libgomp-98b21ff3.so.1",
        "@platforms//os:macos": "lib/libomp.dylib",
        "//conditions:default": "@platforms//:incompatible",
    }),
)
