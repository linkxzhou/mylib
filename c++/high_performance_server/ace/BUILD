package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ace_echo_server",
    srcs = ["ace_echo_server.cpp"],
    copts = [
        "-std=c++17",
        "-O2",
        "-Wall",
    ],
    linkopts = [
        "-lACE",
    ],
    includes = ["/opt/homebrew/include"],
    deps = [],
)