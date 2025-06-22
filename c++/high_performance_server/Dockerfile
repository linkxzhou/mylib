# 使用官方的 CentOS 作为基础镜像
FROM centos:7

# 替换为阿里云镜像源解决网络问题
RUN sed -i 's|^mirrorlist=|#mirrorlist=|g' /etc/yum.repos.d/CentOS-*.repo && \
    sed -i 's|^#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com|g' /etc/yum.repos.d/CentOS-*.repo

# 更新包列表并安装必要的工具和依赖库
RUN yum update -y && yum groupinstall -y "Development Tools" && \
    yum install -y \
    gcc-c++ \
    make \
    cmake \
    git \
    wget \
    openssl-devel \
    zlib-devel \
    libevent-devel \
    libev-devel \
    libuv-devel \
    boost-devel \
    glog-devel \
    gflags-devel \
    double-conversion-devel \
    fmt-devel \
    && yum clean all

# 安装 EPEL 仓库以获取更多包
RUN yum install -y epel-release

# 设置工作目录
WORKDIR /app

# 复制源代码到容器中
COPY . .

# 运行构建脚本生成所有 Makefiles
RUN chmod +x build_all.sh && ./build_all.sh

# 编译所有可用的服务器
RUN make clean && make || true

# 暴露默认端口
EXPOSE 8080

# 设置默认命令 - 运行 libevent 服务器（最兼容的选择）
CMD ["./libevent/libevent_echo_server"]