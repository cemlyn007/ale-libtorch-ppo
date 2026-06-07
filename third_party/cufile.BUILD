load("@rules_cc//cc:defs.bzl", "cc_library")

# cuFile (GPUDirect Storage). In the CUDA toolkit redist, but rules_cuda exposes
# no target for it (registry.bzl: "cufile" -> []). libtorch_cuda.so DT_NEEDEDs
# libcufile.so.0; libcufile_rdma.so.1 ships alongside and resolves libibverbs
# from the host.
cc_library(
    name = "cufile",
    srcs = glob([
        "lib/*.so",
        "lib/*.so.*",
    ]),
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)
