package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "proxygen_echo_server",
    srcs = ["proxygen_echo_server.cpp"],
    copts = [
        "-std=c++17",
        "-O2",
        "-Wall",
    ],
    linkopts = [
        "-lproxygen",
        "-lproxygenlib",
        "-lwangle",
        "-lfolly",
        "-lglog",
        "-lgflags",
        "-ldouble-conversion",
        "-lfmt",
        "-levent",
        "-lssl",
        "-lcrypto",
        "-lpthread",
        "-ldl",
    ],
    includes = ["/opt/homebrew/include"],
    deps = [],
)