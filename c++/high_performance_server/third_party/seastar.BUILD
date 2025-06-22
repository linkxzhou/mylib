cc_library(
    name = "seastar",
    srcs = glob([
        "src/**/*.cc",
        "src/**/*.hh",
    ]),
    hdrs = glob([
        "include/**/*.hh",
        "include/**/*.h",
    ]),
    includes = [
        "include",
        "src",
    ],
    copts = [
        "-std=c++20",
        "-fcoroutines",
        "-DSEASTAR_API_LEVEL=6",
    ],
    linkopts = [
        "-lboost_program_options",
        "-lboost_thread",
        "-lboost_filesystem",
        "-lfmt",
        "-lyaml-cpp",
    ],
    visibility = ["//visibility:public"],
)