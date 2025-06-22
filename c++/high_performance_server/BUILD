package(default_visibility = ["//visibility:public"])

# 构建所有服务器
filegroup(
    name = "all_servers",
    srcs = [
        "//libevent:libevent_echo_server",
        "//libev:libev_echo_server", 
        "//libuv:libuv_echo_server",
        "//boost_asio:boost_asio_echo_server",
        "//ace:ace_echo_server",
        "//seastar:seastar_echo_server",
        "//wangle:wangle_echo_server",
        "//proxygen:proxygen_echo_server",
        "//libco:libco_echo_server",
        "//mongoose:mongoose_echo_server",
    ],
)