load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "ale_sources",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "ale",
    cache_entries = {
        "BUILD_CPP_LIB": "ON",
        "BUILD_PYTHON_LIB": "OFF",
        "BUILD_VECTOR_LIB": "OFF",
        "BUILD_VECTOR_XLA_LIB": "OFF",
        "SDL_SUPPORT": "OFF",
    },
    copts = select({
        "@platforms//os:linux": [
            "-Wno-error=unused-but-set-variable",
            "-Wno-error=unused-variable",
            "-Wno-error=sequence-point",
            "-Wno-error=sign-compare",
        ],
        "@platforms//os:macos": [
            "-Wno-error=unused-private-field",
            "-Wno-inconsistent-missing-override",
        ],
        "//conditions:default": [],
    }),
    generate_args = select({
        "@platforms//os:macos": [
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0",
            "-DCMAKE_AR=/usr/bin/ar",
            "-DCMAKE_RANLIB=/usr/bin/ranlib",
        ],
        "//conditions:default": [],
    }),
    lib_source = ":ale_sources",
    linkopts = select({
        "@platforms//os:linux": [
            "-lpthread",
            "-Wno-stringop-overflow",
        ],
        "@platforms//os:macos": ["-lpthread"],
        "//conditions:default": [],
    }),
    out_static_libs = ["libale.a"],
    visibility = ["//visibility:public"],
    deps = [
        "@zlib",
    ],
)
