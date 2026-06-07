"""Module extension for "configuring" libtorch_bazel."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")

_INTEGRITIES = {
    # Generate with "sha256-$(curl -fsSL "$url" | sha256sum | cut -d' ' -f1 | xxd -r -p | base64)"
    "2.12.0": {
        "linux": "sha256-ozflm2cdMcEOQV4ENeik8s4w7YTpbkf7jTgMlwxF2xI=",
        "macos": "sha256-F3v2xMnpvGyJg702K38QAx5f+r/H+wHh8XOzpNSh81U=",
    },
}

_URLS = {
    "2.12.0": {
        # cu130 (CUDA 13). Unlike <=cu129, the 2.12 zip no longer bundles the
        # CUDA runtime -- it ships only torch's own ~11 .so plus libgomp, and
        # DT_NEEDEDs libcudart/libcublas/libcudnn/libnccl/... by bare soname. We
        # supply those hermetically: the CUDA *toolkit* libs come from rules_cuda
        # @cuda redist (13.0.2), and cudnn/cusparseLt/nccl/nvshmem (not in the
        # toolkit redist) from the bespoke @cudnn/@cusparselt/@nccl/@nvshmem
        # http_archives. See MODULE.bazel + //third_party:cuda_runtime.
        "linux": "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.12.0%2Bcu130.zip",
        "macos": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.12.0.zip",
    },
}

def _libtorch_configure_extension_impl(module_ctx):
    version = "2.12.0"  # default version
    for mod in module_ctx.modules:
        for tag in mod.tags.configure:
            if tag.version:
                version = tag.version

    # Copied from https://skia.googlesource.com/skia/+/9ef295132f0a/bazel/adb_test.bzl.
    if len(HOST_CONSTRAINTS) != 2 or \
       not HOST_CONSTRAINTS[0].startswith("@platforms//cpu:") or \
       not HOST_CONSTRAINTS[1].startswith("@platforms//os:"):
        fail(
            "Expected HOST_CONSTRAINTS to be of the form " +
            """["@platforms//cpu:<cpu>", "@platforms//os:<os>"], got""",
            HOST_CONSTRAINTS,
        )

    # Map the Bazel constants to GOARCH constants. More can be added as needed. See
    # https://github.com/bazelbuild/rules_go/blob/5933b6ed063488472fc14ceca232b3115e8bc39f/go/private/platforms.bzl#LL30C9-L30C9.
    cpu = HOST_CONSTRAINTS[0].removeprefix("@platforms//cpu:")
    os = HOST_CONSTRAINTS[1].removeprefix("@platforms//os:")
    cpu = {
        "x86_64": "amd64",
        "aarch64": "arm64",
    }.get(cpu, cpu)  # Defaults to the original CPU if not in the dictionary.
    os = {
        "osx": "macos",
    }.get(os, os)  # Default to the original OS if not in the dictionary.

    if version not in _URLS or os not in _URLS.get(version, {}):
        fail("PyTorch version %s is not supported for %s" % (version, os))

    http_archive(
        name = "libtorch",
        build_file = "//:third_party/libtorch.BUILD",
        strip_prefix = "libtorch",
        url = _URLS.get(version).get(os),
        integrity = _INTEGRITIES.get(version).get(os),
        # No soname patching needed: the cu130 zip ships only torch's own libs
        # (canonical sonames, no auditwheel hashing), and the hermetic CUDA libs
        # we add separately already carry their soname symlinks.
    )
    return module_ctx.extension_metadata(reproducible = True)

_configure_tag = tag_class(
    attrs = {
        "version": attr.string(doc = "PyTorch version to use"),
    },
)

libtorch_configure_extension = module_extension(
    implementation = _libtorch_configure_extension_impl,
    tag_classes = {
        "configure": _configure_tag,
    },
)
