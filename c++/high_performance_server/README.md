# High Performance Server

这是一个高性能网络服务器实现集合，展示了多种现代 C++ 网络编程框架的使用方法。每个实现都是一个简单的 HTTP Echo 服务器，用于演示不同框架的特性和性能特点。

## 项目概览

本项目包含以下 9 种网络框架的实现：

| 框架 | 类型 | C++ 标准 | 特点 | 代表的开源项目 |
|------|------|----------|------|----------------|
| **libevent** | 事件驱动 | C++11 | 跨平台、轻量级、广泛使用 | Memcached, Tor, Chromium, tmux |
| **libev** | 事件驱动 | C++11 | 高性能、Linux 优化 | Node.js (早期版本), PowerDNS, Varnish |
| **libuv** | 事件驱动 | C++11 | Node.js 底层、跨平台 | Node.js, Julia, Luvit, pyuv |
| **Boost.Asio** | 异步 I/O | C++11 | 功能丰富、标准化 | Beast (HTTP/WebSocket), cpp-netlib, Riak |
| **ACE** | 面向对象 | C++17 | 企业级、模式丰富 | TAO (CORBA), OpenDDS, JAWS Web Server |
| **Seastar** | 无共享 | C++17 | 极高性能、现代设计 | ScyllaDB, Redpanda, Seastar HTTP Server |
| **Wangle** | 异步 | C++17 | Facebook 开源、Pipeline 架构 | Proxygen, McRouter, Facebook Services |
| **Proxygen** | HTTP 专用 | C++17 | Facebook HTTP 库 | Facebook Web Services, Instagram API |
| **Mongoose** | 嵌入式 | C++11 | 轻量级、易集成 | ESP32 项目, IoT 设备, 嵌入式 Web 服务器 |

## 各框架详细架构

## 各框架详细架构

### 1. libevent - 事件驱动架构

**特点**: 
- **跨平台兼容性**: 支持 Linux (epoll)、macOS (kqueue)、Windows (IOCP) 等多种平台
- **多种 I/O 多路复用**: 自动选择最优的 I/O 机制 (epoll/kqueue/select)
- **缓冲事件处理**: 内置缓冲区管理，简化网络编程
- **HTTP 支持**: 内置 HTTP 服务器和客户端功能
- **定时器支持**: 高精度定时器和超时处理
- **线程安全**: 支持多线程环境下的事件处理
- **内存管理**: 高效的内存池和缓冲区管理

**底层架构**:
```mermaid
flowchart TD
    A[Client Socket] --> B[event_base 事件循环]
    B --> C{I/O 多路复用}
    C -->|Linux| D[epoll]
    C -->|macOS| E[kqueue]
    C -->|Windows| F[IOCP]
    C -->|通用| G[select/poll]
    
    D --> H[evconnlistener 监听器]
    E --> H
    F --> H
    G --> H
    
    H --> I[bufferevent 缓冲事件]
    I --> J[读缓冲区]
    I --> K[写缓冲区]
    
    J --> L[echo_read_cb 读回调]
    L --> M[HTTP 解析器]
    M --> N[业务逻辑处理]
    N --> O[响应构建]
    O --> K
    K --> P[bufferevent_write 写回调]
    P --> Q[Client Response]
    
    subgraph "libevent 核心组件"
        B
        H
        I
        L
        P
    end
    
    subgraph "平台适配层"
        D
        E
        F
        G
    end
```

**核心组件**:
- `event_base`: 事件循环核心，管理所有事件
- `evconnlistener`: 连接监听器，处理新连接
- `bufferevent`: 缓冲事件处理，自动管理读写缓冲区
- 回调函数: `echo_read_cb`, `echo_event_cb`
- 平台适配: 自动选择最优 I/O 多路复用机制

### 2. libev - 高性能事件循环

**特点**: 
- **极致性能**: 专为高性能设计，最小化系统调用开销
- **Linux 优化**: 深度优化 epoll 性能，支持 Linux 特有功能
- **轻量级设计**: 代码简洁，内存占用极小
- **多种事件类型**: 支持 I/O、定时器、信号、子进程等事件
- **嵌套事件循环**: 支持事件循环的嵌套调用
- **高精度定时器**: 基于红黑树的高效定时器实现
- **信号处理**: 安全的异步信号处理机制

**底层架构**:
```mermaid
flowchart TD
    A[Client Connection] --> B[ev_loop 主事件循环]
    B --> C[ev_backend 后端选择]
    C -->|最优选择| D[epoll/kqueue]
    C -->|备选| E[poll/select]
    
    D --> F[ev_io accept_watcher]
    E --> F
    F --> G[accept_cb 接受回调]
    G --> H[新连接处理]
    H --> I[ev_io client_watcher]
    
    I --> J[client_cb 客户端回调]
    J --> K[数据读取]
    K --> L[HTTP 协议解析]
    L --> M[业务逻辑]
    M --> N[响应生成]
    N --> O[数据写入]
    O --> P[Client Response]
    
    subgraph "事件监视器类型"
        Q[ev_io I/O事件]
        R[ev_timer 定时器]
        S[ev_signal 信号]
        T[ev_child 子进程]
    end
    
    B --> Q
    B --> R
    B --> S
    B --> T
    
    subgraph "libev 核心"
        B
        F
        I
        G
        J
    end
```

**核心组件**:
- `ev_loop`: 高性能事件循环，支持多种后端
- `ev_io`: I/O 事件监视器，监控文件描述符
- `ev_timer`: 高精度定时器，基于红黑树实现
- `accept_cb`: 连接接受回调，处理新连接
- `client_cb`: 客户端数据处理回调

### 3. libuv - 跨平台异步 I/O

**特点**: 
- **Node.js 底层**: Node.js 的核心依赖，经过大规模生产验证
- **跨平台统一**: 统一的 API 抽象不同平台的异步 I/O
- **线程池**: 内置线程池处理文件 I/O 和 CPU 密集任务
- **异步文件操作**: 完整的异步文件系统 API
- **进程管理**: 跨平台的进程创建和管理
- **网络抽象**: 高级网络 API，支持 TCP、UDP、管道
- **事件循环**: 单线程事件循环 + 多线程工作池

**底层架构**:
```mermaid
flowchart TD
    A[Client Connection] --> B[uv_loop_t 事件循环]
    B --> C{平台检测}
    C -->|Linux| D[epoll]
    C -->|macOS| E[kqueue]
    C -->|Windows| F[IOCP]
    
    D --> G[uv_tcp_t server]
    E --> G
    F --> G
    
    G --> H[uv_listen 监听]
    H --> I[on_new_connection 新连接回调]
    I --> J[uv_tcp_t client]
    J --> K[uv_read_start 开始读取]
    
    K --> L[echo_read 读回调]
    L --> M[数据处理]
    M --> N[HTTP 解析]
    N --> O[业务逻辑]
    O --> P[uv_write 异步写入]
    P --> Q[write_cb 写回调]
    Q --> R[Client Response]
    
    subgraph "线程池 (ThreadPool)"
        S[文件 I/O]
        T[DNS 解析]
        U[CPU 密集任务]
    end
    
    B --> S
    B --> T
    B --> U
    
    subgraph "libuv 核心"
        B
        G
        J
        L
        P
    end
```

**核心组件**:
- `uv_loop_t`: 跨平台事件循环，统一不同平台的异步机制
- `uv_tcp_t`: TCP 句柄，封装网络连接
- `uv_read_start`: 开始异步读取数据
- `uv_write`: 异步写入数据，支持批量写入
- 线程池: 处理阻塞操作，避免阻塞主线程

### 4. Boost.Asio - 异步网络编程

**特点**: 
- **C++ 标准候选**: 设计现代，可能成为 C++ 标准库的一部分
- **类型安全**: 强类型系统，编译时错误检查
- **协程支持**: 支持 C++20 协程，简化异步编程
- **可扩展性**: 支持自定义 I/O 对象和协议
- **SSL/TLS**: 内置 SSL/TLS 支持
- **定时器**: 高精度定时器和截止时间
- **信号处理**: 异步信号处理机制

**底层架构**:
```mermaid
flowchart TD
    A[Client Connection] --> B[io_context I/O上下文]
    B --> C[executor 执行器]
    C --> D{调度策略}
    D -->|单线程| E[strand 串行执行]
    D -->|多线程| F[thread_pool 线程池]
    
    B --> G[tcp::acceptor 接受器]
    G --> H[async_accept 异步接受]
    H --> I[completion_handler 完成处理器]
    I --> J[session 会话对象]
    
    J --> K[tcp::socket 套接字]
    K --> L[async_read_some 异步读取]
    L --> M[read_handler 读处理器]
    M --> N[HTTP 解析]
    N --> O[业务逻辑]
    O --> P[async_write 异步写入]
    P --> Q[write_handler 写处理器]
    Q --> R[Client Response]
    
    subgraph "异步操作链"
        S[async_accept]
        T[async_read_some]
        U[async_write]
        V[async_connect]
    end
    
    subgraph "Boost.Asio 核心"
        B
        G
        J
        K
    end
    
    subgraph "协程支持 (C++20)"
        W[co_await]
        X[awaitable<>]
        Y[use_awaitable]
    end
```

**核心组件**:
- `io_context`: I/O 执行上下文，管理异步操作
- `tcp::acceptor`: TCP 接受器，监听新连接
- `session`: 会话管理类，封装连接生命周期
- 异步操作: `async_accept`, `async_read_some`, `async_write`
- 协程支持: C++20 协程集成

### 5. ACE - 自适应通信环境

**特点**: 
- **企业级框架**: 经过大型企业系统验证的成熟框架
- **设计模式丰富**: 实现了多种网络编程设计模式
- **高度可配置**: 支持编译时和运行时配置
- **跨平台**: 支持 40+ 种操作系统和编译器
- **面向对象**: 完全面向对象的设计
- **组件化**: 模块化设计，可按需使用
- **性能优化**: 针对高并发场景的优化

**底层架构**:
```mermaid
flowchart TD
    A[Client Connection] --> B[ACE_Reactor 反应器]
    B --> C[ACE_Select_Reactor]
    B --> D[ACE_Dev_Poll_Reactor]
    B --> E[ACE_Epoll_Reactor]
    
    C --> F[Accept_Handler 接受处理器]
    D --> F
    E --> F
    
    F --> G[ACE_SOCK_Acceptor 套接字接受器]
    G --> H[handle_input 输入处理]
    H --> I[Echo_Handler 回显处理器]
    
    I --> J[ACE_SOCK_Stream 套接字流]
    J --> K[recv 接收数据]
    K --> L[HTTP 协议解析]
    L --> M[业务逻辑处理]
    M --> N[send_http_response 发送响应]
    N --> O[ACE_SOCK_Stream 输出]
    O --> P[Client Response]
    
    subgraph "设计模式"
        Q[Reactor 反应器]
        R[Proactor 前摄器]
        S[Acceptor-Connector]
        T[Service Configurator]
    end
    
    subgraph "ACE 核心组件"
        B
        F
        I
        G
        J
    end
```

**核心组件**:
- `ACE_Reactor`: 反应器模式核心，支持多种实现
- `ACE_Event_Handler`: 事件处理器基类
- `ACE_SOCK_Acceptor`: 套接字接受器
- `ACE_SOCK_Stream`: 套接字流，封装网络通信
- 设计模式: Reactor、Proactor、Acceptor-Connector

### 6. Seastar - 无共享架构

**特点**: 
- **无共享设计**: 每个 CPU 核心独立运行，避免锁竞争
- **用户态网络栈**: 绕过内核，直接操作网络硬件 (DPDK)
- **协程支持**: 基于 future/promise 的协程模型
- **内存管理**: 自定义内存分配器，减少内存碎片
- **CPU 亲和性**: 严格的 CPU 核心绑定
- **零拷贝**: 最小化数据拷贝操作
- **现代 C++**: 大量使用 C++14/17 特性

**底层架构**:
```mermaid
flowchart TD
    A[Client Connection] --> B[seastar::listen 监听]
    B --> C{CPU 核心分配}
    C --> D[Core 0]
    C --> E[Core 1]
    C --> F[Core N]
    
    D --> G[server_socket 服务器套接字]
    E --> G
    F --> G
    
    G --> H[accept 接受连接]
    H --> I[connected_socket 连接套接字]
    I --> J[input_stream 输入流]
    I --> K[output_stream 输出流]
    
    J --> L[read_exactly 精确读取]
    L --> M[handle_connection 连接处理]
    M --> N[HTTP 解析]
    N --> O[业务逻辑]
    O --> P[create_http_response 创建响应]
    P --> K
    K --> Q[write 写入]
    Q --> R[flush 刷新]
    R --> S[Client Response]
    
    subgraph "Future/Promise 链"
        T[future<>]
        U[do_with]
        V[repeat]
        W[make_ready_future]
        X[when_all]
    end
    
    M --> T
    T --> U
    U --> V
    V --> W
    
    subgraph "无共享架构"
        Y[Per-Core Memory]
        Z[Per-Core Network Queue]
        AA[Per-Core Scheduler]
    end
    
    D --> Y
    D --> Z
    D --> AA
```

**核心组件**:
- `app_template`: 应用程序模板，管理应用生命周期
- `server_socket`: 服务器套接字，支持多核心
- `connected_socket`: 连接套接字，封装网络连接
- `future<>`: 异步操作结果，支持链式调用
- 无共享架构: 每核心独立的内存、网络队列、调度器

### 7. Wangle - Pipeline 架构

**特点**: 
- **Pipeline 设计**: 模块化的请求处理管道
- **Facebook 生产**: Facebook 内部大规模使用
- **类型安全**: 强类型的 Pipeline 组件
- **可组合性**: 灵活的处理器组合
- **协议无关**: 支持多种网络协议
- **负载均衡**: 内置负载均衡和连接池
- **SSL/TLS**: 完整的 SSL/TLS 支持

**底层架构**:
```mermaid
flowchart TD
    A[Client Connection] --> B[ServerBootstrap 服务器引导]
    B --> C[AcceptorFactory 接受器工厂]
    C --> D[Acceptor 接受器]
    D --> E[Pipeline Factory 管道工厂]
    
    E --> F[DefaultPipeline 默认管道]
    F --> G[AsyncSocketHandler 异步套接字处理器]
    G --> H[ByteToMessageDecoder 字节到消息解码器]
    H --> I[HttpDecoder HTTP解码器]
    I --> J[HttpHandler HTTP处理器]
    J --> K[MessageToByteEncoder 消息到字节编码器]
    K --> L[AsyncSocketHandler 输出]
    
    J --> M[业务逻辑处理]
    M --> N[Response Builder 响应构建器]
    N --> O[IOBuf 缓冲区]
    O --> P[writeChain 写入链]
    P --> Q[Client Response]
    
    subgraph "Pipeline 组件"
        R[InboundHandler 入站处理器]
        S[OutboundHandler 出站处理器]
        T[HandlerAdapter 处理器适配器]
        U[Context 上下文]
    end
    
    subgraph "Wangle 核心"
        B
        F
        G
        J
        O
    end
```

**核心组件**:
- `ServerBootstrap`: 服务器引导程序，配置服务器
- `Pipeline`: 处理管道，链式处理请求
- `ByteToMessageDecoder`: 字节到消息解码器
- `HandlerAdapter`: 处理器适配器，连接不同类型的处理器
- `IOBuf`: 高效的缓冲区管理

### 8. Proxygen - HTTP 专用库

**特点**: 
- **HTTP 专用**: 专为 HTTP/1.1 和 HTTP/2 优化
- **Facebook 开源**: Facebook 内部 HTTP 服务的基础
- **HTTP/2 支持**: 完整的 HTTP/2 实现，包括服务器推送
- **流式处理**: 支持大文件的流式上传下载
- **压缩支持**: 内置 gzip、deflate 压缩
- **WebSocket**: 完整的 WebSocket 支持
- **性能监控**: 内置性能指标和监控

**底层架构**:
```mermaid
flowchart TD
    A[HTTP Request] --> B[HTTPServer HTTP服务器]
    B --> C[HTTPSessionAcceptor 会话接受器]
    C --> D[HTTPSession HTTP会话]
    D --> E[HTTPTransaction HTTP事务]
    
    E --> F[RequestHandlerChain 请求处理链]
    F --> G[EchoHandlerFactory 处理器工厂]
    G --> H[EchoHandler 回显处理器]
    
    H --> I[onRequest 请求开始]
    I --> J[onHeaders 头部处理]
    J --> K[onBody 主体处理]
    K --> L[onEOM 消息结束]
    
    L --> M[业务逻辑处理]
    M --> N[ResponseBuilder 响应构建器]
    N --> O[sendWithEOM 发送响应]
    O --> P[HTTP Response]
    
    subgraph "HTTP/2 支持"
        Q[Stream Multiplexing 流复用]
        R[Server Push 服务器推送]
        S[Header Compression 头部压缩]
        T[Flow Control 流量控制]
    end
    
    D --> Q
    D --> R
    D --> S
    D --> T
    
    subgraph "Proxygen 核心"
        B
        D
        E
        H
        N
    end
```

**核心组件**:
- `HTTPServer`: HTTP 服务器，支持 HTTP/1.1 和 HTTP/2
- `RequestHandler`: 请求处理器，处理 HTTP 请求生命周期
- `RequestHandlerFactory`: 处理器工厂，创建请求处理器
- `ResponseBuilder`: 响应构建器，构建 HTTP 响应
- HTTP/2 特性: 流复用、服务器推送、头部压缩

### 9. Mongoose - 嵌入式 Web 服务器

**特点**: 
- **轻量级**: 单文件实现，易于集成
- **嵌入式友好**: 适合资源受限的环境
- **多协议支持**: HTTP、WebSocket、MQTT、CoAP
- **跨平台**: 支持嵌入式系统、桌面、服务器
- **零依赖**: 不依赖外部库
- **事件驱动**: 基于事件的异步处理
- **内置功能**: 文件服务、CGI、SSI 支持

**底层架构**:
```mermaid
flowchart TD
    A[HTTP Request] --> B[mg_mgr 连接管理器]
    B --> C[mg_http_listen HTTP监听]
    C --> D[mg_connection 连接对象]
    D --> E[event_handler 事件处理器]
    
    E --> F{事件类型}
    F -->|连接| G[MG_EV_ACCEPT]
    F -->|HTTP| H[MG_EV_HTTP_MSG]
    F -->|关闭| I[MG_EV_CLOSE]
    F -->|错误| J[MG_EV_ERROR]
    
    H --> K[mg_http_message HTTP消息]
    K --> L[URI 解析]
    L --> M[HTTP 头部解析]
    M --> N[业务逻辑处理]
    N --> O[mg_http_reply HTTP回复]
    O --> P[响应发送]
    P --> Q[Client Response]
    
    subgraph "协议支持"
        R[HTTP/1.1]
        S[WebSocket]
        T[MQTT]
        U[CoAP]
    end
    
    subgraph "Mongoose 核心"
        B
        D
        E
        K
        O
    end
    
    subgraph "嵌入式特性"
        V[单文件实现]
        W[零依赖]
        X[低内存占用]
        Y[实时系统支持]
    end
```

**核心组件**:
- `mg_mgr`: 连接管理器，管理所有网络连接
- `mg_connection`: 连接对象，封装单个网络连接
- `mg_http_listen`: HTTP 监听，启动 HTTP 服务
- `mg_http_reply`: HTTP 响应，发送 HTTP 回复
- 多协议: HTTP、WebSocket、MQTT、CoAP 统一接口

## 性能对比

```mermaid
xychart-beta
    title "Performance Comparison (Requests/sec)"
    x-axis ["libevent", "libev", "libuv", "Boost.Asio", "ACE", "Seastar", "Wangle", "Proxygen", "Mongoose"]
    y-axis "Requests per Second" 0 --> 100000
    bar [25000, 30000, 28000, 22000, 18000, 80000, 35000, 40000, 15000]
```

## 快速开始

### 环境要求

- macOS 或 Linux
- C++11/17 编译器 (GCC 7+ 或 Clang 5+)
- Bazel 6.0+

### 安装 Bazel

```bash
# macOS
brew install bazel

# Ubuntu/Debian
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel

# CentOS/RHEL
sudo dnf copr enable vbatts/bazel
sudo dnf install bazel
```

### 安装依赖

```bash
# macOS
brew install libuv libevent libev boost
brew install folly wangle proxygen  # 可选

# Ubuntu/Debian
sudo apt-get install libuv1-dev libevent-dev libev-dev libboost-all-dev

# CentOS/RHEL
sudo yum install libuv-devel libevent-devel libev-dev boost-devel
```

### 构建项目

```bash
# 构建所有服务器
./build_all_bazel.sh

# 或使用 Bazel 直接构建
bazel build //...

# 构建特定服务器
bazel build //libevent:libevent_echo_server
bazel build //boost_asio:boost_asio_echo_server
bazel build //seastar:seastar_echo_server
bazel build //libev:libev_echo_server
bazel build //libuv:libuv_echo_server
bazel build //mongoose:mongoose_echo_server
```

### 运行服务器

```bash
# 运行 libevent 服务器
bazel run //libevent:libevent_echo_server

# 运行 Seastar 服务器
bazel run //seastar:seastar_echo_server

# 运行 Boost.Asio 服务器
bazel run //boost_asio:boost_asio_echo_server

# 运行 libev 服务器
bazel run //libev:libev_echo_server

# 运行 libuv 服务器
bazel run //libuv:libuv_echo_server

# 运行 Mongoose 服务器
bazel run //mongoose:mongoose_echo_server
```

### 测试服务器

```bash
# 使用 curl 测试
curl -X GET http://localhost:8080/test
```

## Docker 支持

```bash
# 构建镜像
docker build -t high-performance-server .

# 运行容器
docker run -p 8080:8080 high-performance-server

# 使用 docker-compose
docker-compose up
```

## 框架选择指南

### 选择建议

| 场景 | 推荐框架 | 理由 |
|------|----------|------|
| **高并发 Web 服务** | Seastar, Proxygen | 极高性能，现代架构 |
| **跨平台应用** | Boost.Asio, libuv | 标准化，跨平台支持好 |
| **嵌入式系统** | Mongoose, libevent | 轻量级，资源占用少 |
| **企业级应用** | ACE, Boost.Asio | 功能丰富，稳定可靠 |
| **微服务架构** | Wangle, Proxygen | Pipeline 设计，易扩展 |
| **学习研究** | libevent, libev | 代码简洁，易理解 |

### 技术特点对比

```mermaid
quadrantChart
    title Framework Comparison
    x-axis LowComplexity --> HighComplexity
    y-axis LowPerformance --> HighPerformance
    quadrant-1 HighPerformance, HighComplexity
    quadrant-2 HighPerformance, LowComplexity
    quadrant-3 LowPerformance, LowComplexity
    quadrant-4 LowPerformance, HighComplexity
    
    Seastar: [0.8, 0.9]
    Proxygen: [0.7, 0.8]
    Wangle: [0.6, 0.7]
    libev: [0.3, 0.7]
    libuv: [0.4, 0.6]
    Boost.Asio: [0.6, 0.5]
    libevent: [0.3, 0.5]
    ACE: [0.8, 0.4]
    Mongoose: [0.2, 0.3]
```

## 相关资源

- [libevent 官方文档](https://libevent.org/)
- [libev 官方文档](http://software.schmorp.de/pkg/libev.html)
- [libuv 官方文档](https://libuv.org/)
- [Boost.Asio 文档](https://www.boost.org/doc/libs/release/libs/asio/)
- [ACE 官方网站](http://www.dre.vanderbilt.edu/~schmidt/ACE.html)
- [Seastar 官方文档](https://seastar.io/)
- [Wangle GitHub](https://github.com/facebook/wangle)
- [Proxygen GitHub](https://github.com/facebook/proxygen)
- [Mongoose 官方文档](https://mongoose.ws/)
- https://docs.seastar.io/master/tutorial.html
- https://zhuanlan.zhihu.com/p/30738569
- https://blog.csdn.net/Rong_Toa/article/details/108193340
- https://www.cnblogs.com/ahfuzhang/p/15824213.html《Memory Barriers: a Hardware View for Software Hackers》
-