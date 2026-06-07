"""Module extension for "configuring" libtorch_bazel."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")

_INTEGRITIES = {
    # Generate with "sha256-$(curl -fsSL "$url" | sha256sum | cut -d' ' -f1 | xxd -r -p | base64)"
    "2.11.0": {
        "linux": "sha256-KZwJPeN07mAKdFDTsD6vQ4+DjCzKqrvrZelO4QeDBlY=",
        "macos": "sha256-DtwThUXISHkkDMB5lPHpoO01oRAv/iuZ4T86e1ezI+k=",
    },
}

_URLS = {
    "2.11.0": {
        # cu126: PyTorch dropped the pre-cxx11 ABI, so the Linux artifact lost the
        # "cxx11-abi-" filename prefix. (2.12 onward unbundles the CUDA runtime
        # from the zip; 2.11 is the last release that still ships it inline.)
        "linux": "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.11.0%2Bcu126.zip",
        "macos": "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.11.0.zip",
    },
}

def _libtorch_configure_extension_impl(module_ctx):
    version = "2.11.0"  # default version
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
        # The vendored CUDA/OpenMP libs have auditwheel-hashed filenames (e.g.
        # libcudart-e6b31d9c.so.12) but canonical sonames (libcudart.so.12).
        # Anything we link records a DT_NEEDED on the canonical name, which
        # exists nowhere in the distribution -- recreate the soname -> hashed-file
        # symlinks (as a normal CUDA install would) so those entries resolve.
        patch_cmds = [
            """
            for f in lib/*.so*; do
                b=$(basename "$f")
                [ -L "lib/$b" ] && continue
                canon=$(printf '%s' "$b" | sed -E 's/-[0-9a-f]+(\\.so)/\\1/')
                if [ "$canon" != "$b" ] && [ ! -e "lib/$canon" ]; then
                    ln -s "$b" "lib/$canon"
                fi
            done
            """,
        ],
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
