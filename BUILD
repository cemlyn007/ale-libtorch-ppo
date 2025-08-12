load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//src:all": "",
        "//test/ai:all": "",
    },
)

filegroup(
    name = "roms",
    srcs = glob(["roms/**/*.bin"]),
    visibility = ["//visibility:public"],
)
