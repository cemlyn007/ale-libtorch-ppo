load("@rules_cc//cc:defs.bzl", "cc_test")

googletest_copts = [
    "-Iexternal/gtest/googletest/include",
    "-Iexternal/gtest/googletest",
]

# Pure orchestration/config logic — no GPU, no training stack — so it runs
# anywhere. Guards the batch-geometry coupling: every sampled arm must be
# validate()-clean, and mini_batch_size must stay derived.
cc_test(
    name = "bandit_test",
    srcs = ["bandit_test.cc"],
    copts = googletest_copts,
    deps = [
        "//src/training:bandit",
        "//src/training:config",
        "@googletest//:gtest_main",
    ],
)
