load("@rules_cc//cc:defs.bzl", "cc_library")

# cuDNN 9 (CUDA 13 redist). libtorch_cuda.so DT_NEEDEDs libcudnn.so.9, which in
# turn dlopen's the libcudnn_engines_*/ops/cnn/... sub-libs. Globbing them all
# into one cc_library co-locates them in a single solib dir so cuDNN's $ORIGIN
# dlopen finds its siblings at runtime.
cc_library(
    name = "cudnn",
    srcs = glob([
        "lib/*.so",
        "lib/*.so.*",
    ]),
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)
