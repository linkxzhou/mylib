# Docker 使用指南

本项目提供了 Docker 支持，方便在 Mac 上使用 Linux 环境编译和运行 C++ 并发服务器。

## 快速开始

### 1. 构建 Docker 镜像

```bash
docker build -t concurrency-server .
```

### 2. 运行容器

```bash
# 使用默认的 thread_pool 模型
docker run -p 8080:8080 concurrency-server

# 或者指定其他服务器模型
docker run -p 8080:8080 concurrency-server ./concurrency_server epoll 8080
```

### 3. 使用 Docker Compose

```bash
# 启动默认的 thread_pool 服务器
docker-compose up

# 启动特定的服务器模型
docker-compose --profile epoll up epoll-server
docker-compose --profile reactor up reactor-server
docker-compose --profile coroutine up coroutine-server
```

## 可用的服务器模型

- `single_process` - 单进程服务器
- `multi_process` - 多进程服务器
- `multi_thread` - 多线程服务器
- `process_pool1` - 进程池服务器（共享套接字）
- `process_pool2` - 进程池服务器（SO_REUSEPORT）
- `thread_pool` - 线程池服务器
- `leader_follower` - Leader-Follower 服务器
- `select` - Select I/O 多路复用服务器
- `poll` - Poll I/O 多路复用服务器
- `epoll` - Epoll I/O 多路复用服务器（Linux）
- `kqueue` - Kqueue I/O 多路复用服务器（BSD/macOS）
- `reactor` - Reactor 模式服务器
- `coroutine` - 协程风格服务器
- `event_loop` - 事件循环服务器
- `work_stealing` - 工作窃取服务器
- `actor` - Actor 模型服务器
- `fiber` - 纤程/绿色线程服务器
- `producer_consumer` - 生产者-消费者服务器
- `half_sync_async` - 半同步/半异步服务器
- `proactor` - Proactor 模式服务器
- `pipeline` - 管道服务器
- `hybrid` - 混合模型服务器

## 测试服务器

### 使用 curl 测试

```bash
curl http://localhost:8080/
```

### 使用 Apache Bench 进行性能测试

```bash
# 在容器内运行基准测试
docker exec -it <container_id> make bench

# 或者从主机运行
ab -n 1000 -c 10 http://localhost:8080/
```

## 开发模式

如果你想在开发过程中修改代码并重新编译，可以挂载源代码目录：

```bash
docker run -it -v $(pwd):/app -p 8080:8080 ubuntu:22.04 bash
# 在容器内
apt-get update && apt-get install -y build-essential g++ make apache2-utils
cd /app
make clean && make
./concurrency_server thread_pool 8080
```

## 多架构支持

如果需要构建支持多架构的镜像（如 ARM64 和 AMD64）：

```bash
# 创建并使用 buildx builder
docker buildx create --use

# 构建多架构镜像
docker buildx build --platform linux/amd64,linux/arm64 -t concurrency-server:latest .
```

## 故障排除

### 端口冲突
如果端口 8080 已被占用，可以映射到其他端口：

```bash
docker run -p 8081:8080 concurrency-server
```

### 查看容器日志

```bash
docker logs <container_id>
```

### 进入容器调试

```bash
docker exec -it <container_id> bash
```