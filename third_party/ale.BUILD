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
    copts = [
        "-pthread",
        "-Wno-error=unused-but-set-variable",
        "-Wno-error=unused-variable",
        "-Wno-error=sequence-point",
        "-Wno-error=sign-compare",
        # MacOS
        "-Wno-error=unused-private-field",
        "-Wno-inconsistent-missing-override",
    ],
    generate_args = [
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0",
        "-DCMAKE_AR=/usr/bin/ar",
        "-DCMAKE_RANLIB=/usr/bin/ranlib",
    ],
    lib_source = ":ale_sources",
    linkopts = ["-lpthread"],
    out_static_libs = ["libale.a"],
    visibility = ["//visibility:public"],
)
