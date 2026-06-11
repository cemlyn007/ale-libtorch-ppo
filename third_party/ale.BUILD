load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "ale_sources",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

_CACHE_ENTRIES = {
    "BUILD_CPP_LIB": "ON",
    "BUILD_PYTHON_LIB": "OFF",
    "BUILD_VECTOR_LIB": "OFF",
    "BUILD_VECTOR_XLA_LIB": "OFF",
    "SDL_SUPPORT": "OFF",
}

# PGO for the emulator hot loop (~+10% stepping throughput), selected via
# @//third_party:ale_pgo; scripts/ale_pgo.sh drives the gen->run->use cycle.
# Flags go in CMAKE_CXX_FLAGS_RELEASE: ordered after the Bazel toolchain's
# CXXFLAGS, so -fomit-frame-pointer overrides its -fno-omit-frame-pointer.
_OPT_FLAGS = "-O3 -DNDEBUG -fomit-frame-pointer"

# -fprofile-prefix-path strips the build dir from the object paths that name
# the .gcda files, so the profile filenames stay stable across builds. GCC
# also hashes the absolute SOURCE path into each function's profile checksum
# and no -f*-prefix-map flag covers that hash, so gen/use must additionally
# compile from identical source paths: the ale-pgo-* configs disable
# sandboxing for this action (see .bazelrc) to pin them to the execroot.
# $$VAR (one trailing $$ would leak a literal `$` that make then mangles)
# becomes $VAR in the foreign_cc build script, expanded by bash at configure
# time.
_PGO_COMMON_FLAGS = " -fprofile-prefix-path=$$BUILD_TMPDIR"

_PGO_GEN_FLAGS = (_OPT_FLAGS + _PGO_COMMON_FLAGS +
                  " -fprofile-generate=/tmp/ale-bazel-pgo")

# -fprofile-correction: profiles come from multithreaded runs with racy
# counter updates. Missing profiles (carts the run never loaded) only warn.
_PGO_USE_FLAGS = (_OPT_FLAGS + _PGO_COMMON_FLAGS +
                  " -fprofile-use=$$EXT_BUILD_ROOT/third_party/ale-pgo" +
                  " -fprofile-correction -Wno-error=missing-profile")

cmake(
    name = "ale",
    build_data = select({
        "@//third_party:ale_pgo_use": ["@//third_party:ale_pgo_profiles"],
        "//conditions:default": [],
    }),
    cache_entries = select({
        "@//third_party:ale_pgo_gen": dict(_CACHE_ENTRIES, CMAKE_CXX_FLAGS_RELEASE = _PGO_GEN_FLAGS),
        "@//third_party:ale_pgo_use": dict(_CACHE_ENTRIES, CMAKE_CXX_FLAGS_RELEASE = _PGO_USE_FLAGS),
        "//conditions:default": _CACHE_ENTRIES,
    }),
    copts = select({
        "@platforms//os:linux": [
            "-Wno-error=unused-but-set-variable",
            "-Wno-error=unused-variable",
            "-Wno-error=sequence-point",
            "-Wno-error=sign-compare",
        ],
        "@platforms//os:macos": [
            "-Wno-error=unused-but-set-variable",
            "-Wno-error=unused-variable",
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
    }) + select({
        # The instrumented archive needs libgcov in the consuming binary.
        "@//third_party:ale_pgo_gen": ["-fprofile-generate"],
        "//conditions:default": [],
    }),
    out_static_libs = ["libale.a"],
    visibility = ["//visibility:public"],
    deps = [
        "@zlib",
    ],
)
