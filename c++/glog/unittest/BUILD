package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gtest",
    srcs = [
        "@gtest//:src/gtest-all.cc",
    ],
    hdrs = [
        "@gtest//:include/gtest/gtest.h",
    ],
    includes = [
        "@gtest//:include",
    ],
    visibility = ["//visibility:public"],
    linkstatic = True,
)

cc_test(
    name = "demangle_unittest",
    srcs = glob([
        "demangle_unittest.cpp",
    ]),
    deps = [
        "//glog:glog",
        "@gtest//:gtest",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare",
    ],
)

cc_test(
    name = "signalhandler_unittest",
    srcs = glob([
        "signalhandler_unittest.cpp",
    ]),
    deps = [
        "//glog:glog",
        "@gtest//:gtest",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare",
    ],
)

cc_test(
    name = "symbolize_unittest",
    srcs = glob([
        "symbolize_unittest.cpp",
    ]),
    deps = [
        "//glog:glog",
        "@gtest//:gtest",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare",
    ],
)

cc_test(
    name = "utilities_unittest",
    srcs = glob([
        "utilities_unittest.cpp",
    ]),
    deps = [
        "//glog:glog",
        "@gtest//:gtest",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare",
    ],
)

cc_test(
    name = "stl_logging_unittest",
    srcs = glob([
        "stl_logging_unittest.cpp",
    ]),
    deps = [
        "//glog:glog",
        "@gtest//:gtest",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare",
    ],
)