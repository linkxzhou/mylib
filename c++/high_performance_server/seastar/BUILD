package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "seastar_echo_server",
    srcs = ["seastar_echo_server.cpp"],
    copts = [
        "-std=c++17",
        "-O2",
        "-Wall",
    ],
    linkopts = [
        "-lseastar",
    ],
    deps = [],
)