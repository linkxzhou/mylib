package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "boost_asio_echo_server",
    srcs = ["boost_asio_echo_server.cpp"],
    copts = [
        "-std=c++11",
        "-O2",
        "-Wall",
    ],
    deps = [
        "//third_party:boost_system",
    ],
    linkopts = ["-lpthread"],
)