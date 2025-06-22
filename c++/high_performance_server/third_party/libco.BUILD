package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libco",
    srcs = [
        "co_epoll.cpp",
        "co_hook_sys_call.cpp",
        "co_routine.cpp",
        "coctx.cpp",
        "coctx_swap.S",
    ],
    hdrs = [
        "co_epoll.h",
        "co_routine.h",
        "co_routine_inner.h",
        "co_routine_specific.h",  # 添加缺失的头文件
        "coctx.h",
    ],
    includes = ["."],
    copts = [
        "-std=c++11",
        "-O2",
        "-Wall",
        "-fPIC",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
)