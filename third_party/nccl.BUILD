load("@rules_cc//cc:defs.bzl", "cc_library")

# NCCL (from the nvidia-nccl-cu13 wheel -- NVIDIA ships no redist tarball).
# libtorch_cuda.so DT_NEEDEDs libnccl.so.2; this single-GPU app never calls
# collectives, but the soname must resolve for the binary to load.
cc_library(
    name = "nccl",
    srcs = glob(["nvidia/nccl/lib/*.so*"]),
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)
