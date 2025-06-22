package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libco_echo_server",
    srcs = ["libco_echo_server.cpp"],
    copts = [
        "-std=c++11",
        "-O2",
        "-Wall",
    ],
    deps = [
        "@libco//:libco",
    ],
)