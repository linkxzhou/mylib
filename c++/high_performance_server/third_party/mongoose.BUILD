package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mongoose",
    srcs = ["mongoose.c"],
    hdrs = ["mongoose.h"],
    includes = ["."],
    copts = [
        "-std=c99",
        "-O2",
        "-Wall",
        "-DMG_ENABLE_LINES=1",
    ],
    visibility = ["//visibility:public"],
)