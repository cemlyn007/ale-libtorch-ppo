load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

# libtorch is a prebuilt distribution whose shared libraries are linked with
# RPATH=$ORIGIN and form one connected dependency web: libtorch_cuda.so needs
# libtorch_cpu.so, which needs libc10.so and libgomp, and the CUDA libraries
# (libcudart, libcublas, cuDNN, NVTX, ...) sit alongside them. Because
# DT_RUNPATH is NOT transitive, every one of these libraries can only find its
# siblings in its OWN directory -- so they must all live together at runtime.
#
# Bazel co-locates precompiled shared libraries in a single _solib directory
# only when they belong to the SAME target's srcs. Splitting them across
# separate cc_import targets (one _solib dir each) or a data filegroup (a
# separate runfiles dir) breaks $ORIGIN resolution. We therefore keep every
# loadable libtorch library in one cc_library.
#
# We deliberately drop:
#   * the static archives (*.a) -- not needed by a shared-library app,
#   * libtorch_python.so          -- the Python bindings, and
#   * lib*_test.so                -- libtorch's bundled unit-test backends,
# none of which an application should link.
#
# The vendored CUDA/OpenMP libraries use auditwheel-hashed filenames whose
# DT_SONAME is canonical (libcudart-<hash>.so.12 has soname libcudart.so.12).
# Linking them makes our binary record a DT_NEEDED on the canonical soname; the
# repository rule (third_party/libtorch.bzl) recreates the canonical
# soname -> hashed-file symlinks so those entries resolve at runtime, exactly as
# a normal CUDA install would provide them. The glob picks up both the hashed
# files (which libtorch's own libraries reference) and the symlinks.

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
