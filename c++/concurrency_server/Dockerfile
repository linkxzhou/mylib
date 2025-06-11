# 使用官方的 CentOS 作为基础镜像
FROM centos:7

# 替换为阿里云镜像源解决网络问题
RUN sed -i 's|^mirrorlist=|#mirrorlist=|g' /etc/yum.repos.d/CentOS-*.repo && \
    sed -i 's|^#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com|g' /etc/yum.repos.d/CentOS-*.repo

# 更新包列表并安装必要的工具
RUN yum update -y && yum groupinstall -y "Development Tools" && \
    yum install -y \
    gcc-c++ \
    make \
    httpd-tools \
    && yum clean all

# 设置工作目录
WORKDIR /app

# 复制源代码到容器中
COPY . .

# 编译项目
RUN make clean && make

# 暴露默认端口
EXPOSE 8080

# 设置默认命令
CMD ["./concurrency_server", "coroutine", "8080"]