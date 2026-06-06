load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

# libtorch's prebuilt shared libraries are linked with RPATH=$ORIGIN and find
# each other (and the vendored CUDA/OpenMP libs) by sibling lookup. Since RPATH
# isn't transitive, they must all share one runtime directory -- which Bazel
# only guarantees when they're srcs of a single target. So every loadable lib
# lives in this one cc_library; splitting into cc_imports or a data filegroup
# would scatter them and break $ORIGIN resolution.
#
# The glob excludes what an app shouldn't link (static archives, the Python
# bindings, the bundled test backends). The vendored libs have auditwheel-hashed
# filenames but canonical sonames; libtorch.bzl recreates the canonical-soname
# symlinks so our DT_NEEDED entries resolve.

cc_library(
    name = "torch",
    srcs = select({
        "@platforms//os:linux": glob(
            [
                "lib/*.so",
                "lib/*.so.*",
            ],
            exclude = [
                "lib/libtorch_python.so",
                "lib/lib*_test.so",
                # Android NNAPI backend; pulls in undefined CPython symbols.
                "lib/libnnapi_backend.so",
                # NVRTC's runtime-compiled builtins; dlopen'd by libnvrtc, not
                # linked.
                "lib/libnvrtc-builtins.so",
            ],
            # Empty on a macOS checkout; globs are evaluated regardless of the
            # surrounding select, so allow no matches there.
            allow_empty = True,
        ),
        "@platforms//os:macos": glob(["lib/*.dylib"], allow_empty = True),
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    hdrs = glob([
        "include/c10/**",
        "include/torch/**",
        "include/ATen/**",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
)
