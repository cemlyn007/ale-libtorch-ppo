load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Compile-time CUDA headers reachable from train.h (cuda_runtime.h, cublas_v2.h,
# cusolverDn.h, ...). We can't use @cuda//:cuda_headers: it blindly deps
# culibos_headers, but the culibos component (needed for cudart's culibos_a on
# CUDA 13) ships no headers target, so the aggregate fails to analyze. Compose
# the per-component header targets ourselves instead.
cc_library(
    name = "cuda_headers",
    target_compatible_with = ["@platforms//os:linux"],
    deps = [
        "@cuda//:cccl_headers",
        "@cuda//:crt_headers",  # crt/host_config.h, pulled by cuda_runtime.h
        "@cuda//:cublas_headers",
        "@cuda//:cudart_headers",
        "@cuda//:cufft_headers",
        "@cuda//:cupti_headers",
        "@cuda//:curand_headers",
        "@cuda//:cusolver_headers",
        "@cuda//:cusparse_headers",
        "@cuda//:nvjitlink_headers",
        "@cuda//:nvrtc_headers",
        "@cuda//:nvtx_headers",
    ],
)

# The CUDA runtime libraries that libtorch's cu130 .so DT_NEEDED but no longer
# bundles. Toolkit libs come from rules_cuda's @cuda redist (13.0.2); cufile,
# cudnn, cusparseLt, nccl and nvshmem are fetched separately (see MODULE.bazel).
# Depend on this from any binary that links @libtorch//:torch on Linux. Resolution
# across these separate solib dirs relies on -Wl,--disable-new-dtags (transitive
# DT_RPATH) -- see the train binary's linkopts.
cc_library(
    name = "cuda_runtime",
    # The cu130 libtorch .so find their CUDA siblings (libcudart, libcublas, ...)
    # by bare soname, but those live in separate solib dirs. DT_RUNPATH isn't
    # transitive, so emit old-style (transitive) DT_RPATH instead. Propagates to
    # any binary that links this library.
    linkopts = ["-Wl,--disable-new-dtags"],
    target_compatible_with = ["@platforms//os:linux"],
    deps = [
        "@cuda//:cublas",  # libcublas.so.13 + libcublasLt.so.13
        "@cuda//:cuda_runtime",  # libcudart.so.13
        "@cuda//:cufft",  # libcufft.so.12
        "@cuda//:cupti",  # libcupti.so.13
        "@cuda//:curand",  # libcurand.so.10
        "@cuda//:cusolver",  # libcusolver.so.12
        "@cuda//:cusparse",  # libcusparse.so.12
        "@cuda//:nvjitlink",  # libnvJitLink.so.13 (cublasLt dep)
        "@cuda//:nvrtc",  # libnvrtc.so.13
        "@cudnn",  # libcudnn.so.9 (+ dlopen'd engine sub-libs)
        "@cufile",  # libcufile.so.0
        "@cusparselt",  # libcusparseLt.so.0
        "@nccl",  # libnccl.so.2
        "@nvshmem",  # libnvshmem_host.so.3
    ],
)
