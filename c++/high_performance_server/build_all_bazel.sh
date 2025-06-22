#!/bin/bash

# Bazel 构建所有网络框架测试用例的脚本

echo "Building all network framework test cases with Bazel (Bzlmod + GitHub deps)..."

# 检查 Bazel 是否安装
if ! command -v bazel &> /dev/null; then
    echo "Error: Bazel is not installed. Please install Bazel first."
    echo "Visit: https://bazel.build/install"
    exit 1
fi

# 检查 Bazel 版本
BAZEL_VERSION=$(bazel version | grep "Build label" | cut -d' ' -f3)
echo "Using Bazel version: $BAZEL_VERSION"

# 清理之前的构建
echo "Cleaning previous builds..."
bazel clean

# 下载并构建依赖
echo "Fetching GitHub dependencies..."
bazel fetch //...

# 构建所有目标
echo "Building all servers with GitHub dependencies..."
bazel build //... --enable_bzlmod

if [ $? -eq 0 ]; then
    echo "All servers built successfully!"
    echo ""
    echo "Built executables are located in:"
    echo "  bazel-bin/libevent/libevent_echo_server"
    echo "  bazel-bin/libev/libev_echo_server"
    echo "  bazel-bin/libuv/libuv_echo_server"
    echo "  bazel-bin/boost_asio/boost_asio_echo_server"
    echo "  bazel-bin/ace/ace_echo_server"
    echo "  bazel-bin/seastar/seastar_echo_server"
    echo "  bazel-bin/wangle/wangle_echo_server"
    echo "  bazel-bin/proxygen/proxygen_echo_server"
    echo "  bazel-bin/libco/libco_echo_server"
    echo "  bazel-bin/mongoose/mongoose_echo_server"
    echo ""
    echo "To run a specific server:"
    echo "  bazel run //libevent:libevent_echo_server"
    echo "  bazel run //libco:libco_echo_server"
    echo "  # ... etc"
else
    echo "Build failed. Please check the error messages above."
    echo "For system libraries, make sure they are installed:"
    echo "  brew install boost ace seastar"
    echo "  brew install folly wangle proxygen (if available)"
    exit 1
fi