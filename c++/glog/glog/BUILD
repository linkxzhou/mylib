package(default_visibility = ["//visibility:public"])

# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_library(
    name = "glog",
    srcs = glob([
        "*.cpp",
    ]),
    hdrs = glob(
        [
            "*.h",
        ]
    ),
    deps = [
        "//base:base",
    ],
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare",
        "-Wno-nullability-completeness",
        "-Wno-expansion-to-defined",
        "-DHAVE_SYS_SYSCALL_H",  # macOS uses sys/syscall.h instead of syscall.h
        "-DHAVE_DLADDR",         # macOS has dladdr support
        "-DOS_MACOSX",           # Define macOS platform
    ],
)