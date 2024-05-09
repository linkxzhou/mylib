package(default_visibility = ["//visibility:public"])

cc_library(
    name = "base",
    srcs = glob([
        "*.cpp",
    ]),
    hdrs = glob(
        [
            "*.h",
        ]
    ),
    copts = [
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare",
    ],
)