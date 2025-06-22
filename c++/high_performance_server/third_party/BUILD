package(default_visibility = ["//visibility:public"])

# 系统库的虚拟目标（用于不从 GitHub 下载的库）
cc_library(
    name = "boost_system",
    linkopts = ["-lboost_system"],
)

cc_library(
    name = "ace",
    linkopts = ["-lACE"],
)

cc_library(
    name = "seastar",
    linkopts = ["-lseastar"],
)

cc_library(
    name = "wangle",
    linkopts = [
        "-lwangle",
        "-lfolly",
        "-lglog",
        "-lgflags",
        "-ldouble-conversion",
    ],
)

cc_library(
    name = "proxygen",
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
)

cc_library(
    name = "libco",
    linkopts = [
        "-lcolib",
        "-ldl",
    ],
)