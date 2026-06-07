load("@rules_cc//cc:defs.bzl", "cc_library")

# NVSHMEM (CUDA 13 redist). torch's own libtorch_nvshmem.so DT_NEEDEDs
# libnvshmem_host.so.3, so it must be present for the binary to load. We link
# ONLY the host lib: the transport/bootstrap plugins (ibgda/ucx/libfabric/...)
# DT_NEEDED libfabric/ibverbs that aren't present, and would break eager load.
# They're dlopen-only and unused on the single-GPU path, so we omit them.
cc_library(
    name = "nvshmem",
    srcs = glob(["lib/libnvshmem_host.so*"]),
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)
