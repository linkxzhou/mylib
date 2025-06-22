package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libuv_echo_server",
    srcs = ["libuv_echo_server.cpp"],
    copts = [
        "-std=c++11",
        "-O2",
        "-Wall",
    ],
    deps = [
        "@libuv//:libuv",
    ],
)