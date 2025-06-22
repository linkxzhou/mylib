package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "wangle_echo_server",
    srcs = ["wangle_echo_server.cpp"],
    copts = [
        "-std=c++17",
        "-O2",
        "-Wall",
    ],
    linkopts = [
        "-lwangle",
        "-lfolly",
        "-lglog",
        "-lgflags",
        "-ldouble-conversion",
    ],
    includes = ["/opt/homebrew/include"],
    deps = [],
)