package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mongoose_echo_server",
    srcs = [
        "mongoose_echo_server.cpp",
    ],
    copts = [
        "-std=c++11",
        "-O2",
        "-Wall",
        "-DMG_ENABLE_LINES",
    ],
    deps = [
        "@mongoose//:mongoose",
    ],
)