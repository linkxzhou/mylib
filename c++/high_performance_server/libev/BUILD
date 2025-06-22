package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libev_echo_server",
    srcs = ["libev_echo_server.cpp"],
    copts = [
        "-std=c++11",
        "-O2",
        "-Wall",
    ],
    deps = [
        "@libev//:libev",
    ],
)