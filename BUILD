load("@aspect_rules_lint//format:defs.bzl", "format_multirun")
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

# `bazel run //:format` to fix, `//:format.check` to verify. clang-format only;
# style lives in .clang-format. `cuda` covers future .cu/.cuh sources.
format_multirun(
    name = "format",
    c = "@llvm_toolchain_llvm//:bin/clang-format",
    cc = "@llvm_toolchain_llvm//:bin/clang-format",
    cuda = "@llvm_toolchain_llvm//:bin/clang-format",
)

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//src:all": "",
        "//src/ai:all": "",
        "//src/ai/ppo:all": "",
        "//src/bin:all": "",
        "//src/training:all": "",
        "//test/ai:all": "",
    },
)

filegroup(
    name = "roms",
    srcs = glob(["roms/**/*.bin"]),
    visibility = ["//visibility:public"],
)
