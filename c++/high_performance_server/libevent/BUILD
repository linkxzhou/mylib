package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libevent_echo_server",
    srcs = ["libevent_echo_server.cpp"],
    copts = [
        "-std=c++11",
        "-O2",
        "-Wall",
    ],
    deps = [
        "@libevent//:libevent",
    ],
)