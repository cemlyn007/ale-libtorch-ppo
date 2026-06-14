load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

package(default_visibility = ["//visibility:public"])

# PGO mode for the ALE emulator build: `gen` instruments it (profiles land in
# /tmp/ale-bazel-pgo on clean process exit), `use` feeds back the profiles in
# third_party/ale-pgo/. Drive the whole cycle with scripts/ale_pgo.sh.
string_flag(
    name = "ale_pgo",
    build_setting_default = "off",
    values = [
        "off",
        "gen",
        "use",
    ],
)

config_setting(
    name = "ale_pgo_gen",
    flag_values = {":ale_pgo": "gen"},
)

config_setting(
    name = "ale_pgo_use",
    flag_values = {":ale_pgo": "use"},
)

# Harvested .gcda profiles (machine- and GCC-version-specific, gitignored).
# Declared as an input of the `use` build so a profile refresh re-runs cmake.
filegroup(
    name = "ale_pgo_profiles",
    srcs = glob(
        ["ale-pgo/**"],
        allow_empty = True,
    ),
)
