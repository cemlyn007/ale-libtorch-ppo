load("@rules_cc//cc:defs.bzl", "cc_library")

# Native Bazel build of ALE's C++ core, replacing the upstream CMake build.
# Scope matches the old cmake() config -- C++ lib only, SDL/vector/python off
# (the SDL-named sources are internally #ifdef'd to stubs). The srcs globs
# cover the target_sources lists in src/ale/*/CMakeLists.txt exactly.

# configure_file(version.hpp.in) equivalent. The release tarball has no git
# metadata, so the SHA is upstream's "unknown" fallback.
genrule(
    name = "version_hpp",
    srcs = ["version.txt"],
    outs = ["src/ale/version.hpp"],
    cmd = """v=$$(tr -d '[:space:]' < $<); IFS=. read -r major minor patch <<<"$$v"; cat > $@ <<EOF
#ifndef __VERSION_HPP__
#define __VERSION_HPP__

#define ALE_VERSION "$$v"
#define ALE_VERSION_MAJOR "$$major"
#define ALE_VERSION_MINOR "$$minor"
#define ALE_VERSION_PATCH "$$patch"
#define ALE_VERSION_GIT_SHA "unknown"

// This isn't entirely accurate, there's been
// some changes post 2.4.2.
#define STELLA_VERSION "2.4.2"

#endif // __VERSION_HPP__
EOF""",
)

# Match upstream's Release build in every -c mode, plus quiet its warnings
# (the emulator code trips -Wall; severity only, our -Werror exempts external/).
_BASE_COPTS = [
    "-O3",
    "-DNDEBUG",
    "-Wno-unused-but-set-variable",
    "-Wno-unused-variable",
    "-Wno-sequence-point",
    "-Wno-sign-compare",
    "-Wno-stringop-overflow",
]

# PGO for the emulator hot loop (~+10% stepping throughput), selected via
# @//third_party:ale_pgo; scripts/ale_pgo.sh drives the gen->run->use cycle.
# Sandboxing is no obstacle: Bazel compiles with PWD=/proc/self/cwd, so the
# cwd GCC bakes into each .gcda name (and resolves the relative -fprofile-use
# dir against) is identical across sandboxes and builds.
_PGO_GEN_COPTS = [
    "-fomit-frame-pointer",
    "-fprofile-generate=/tmp/ale-bazel-pgo",
]

# -fprofile-correction: profiles come from multithreaded runs with racy
# counter updates. Missing profiles (carts the run never loaded) only warn.
_PGO_USE_COPTS = [
    "-fomit-frame-pointer",
    "-fprofile-use=third_party/ale-pgo",
    "-fprofile-correction",
    "-Wno-missing-profile",
]

cc_library(
    name = "ale",
    srcs = glob([
        "src/ale/common/*.cpp",
        "src/ale/common/*.cxx",
        "src/ale/emucore/*.cxx",
        "src/ale/environment/*.cpp",
        "src/ale/games/*.cpp",
        "src/ale/games/supported/*.cpp",
    ]) + ["src/ale/ale_interface.cpp"],
    hdrs = glob(
        [
            "src/ale/**/*.h",
            "src/ale/**/*.hpp",
            "src/ale/**/*.hxx",
        ],
        exclude = [
            "src/ale/external/**",
            "src/ale/python/**",
            "src/ale/vector/**",
        ],
        allow_empty = True,
    ) + ["src/ale/version.hpp"],
    # M6502 instruction tables, #included mid-class by the CPU cores.
    textual_hdrs = glob(["src/ale/emucore/*.ins"]),
    copts = _BASE_COPTS + select({
        "@//third_party:ale_pgo_gen": _PGO_GEN_COPTS,
        "@//third_party:ale_pgo_use": _PGO_USE_COPTS,
        "//conditions:default": [],
    }),
    additional_compiler_inputs = select({
        "@//third_party:ale_pgo_use": ["@//third_party:ale_pgo_profiles"],
        "//conditions:default": [],
    }),
    # src: internal + consumer `ale/...` includes; src/ale: ale_interface.hpp's
    # quote-include of the generated version.hpp.
    includes = [
        "src",
        "src/ale",
    ],
    linkopts = ["-lpthread"] + select({
        # The instrumented objects need libgcov in the consuming binary.
        "@//third_party:ale_pgo_gen": ["-fprofile-generate"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = ["@zlib"],
)
