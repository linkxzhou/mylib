#include "enhanced_event_loop.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <algorithm>

// EnhancedEventLoop 实现
EnhancedEventLoop::EnhancedEventLoop(EventDispatcherFactory::Type dispatcher_type, size_t max_connections)
    : listen_fd_(-1), port_(0), running_(false), keepalive_timeout_(std::chrono::seconds(300)) {
    
    dispatcher_ = EventDispatcherFactory::create(dispatcher_type);
    if (!dispatcher_) {
        throw std::runtime_error("Failed to create event dispatcher");
    }
    
    conn_manager_ = std::make_unique<ConnectionManager>(max_connections);
    
    // 设置连接管理器回调
    conn_manager_->set_new_connection_callback(
        [this](std::shared_ptr<Connection> conn) {
            on_new_connection(conn);
        });
    
    conn_manager_->set_data_received_callback(
        [this](std::shared_ptr<Connection> conn, const char* data, size_t size) {
            on_message_received(conn, data, size);
        });
    
    conn_manager_->set_connection_closed_callback(
        [this](std::shared_ptr<Connection> conn) {
            on_connection_closed(conn);
        });
    
    conn_manager_->set_error_callback(
        [this](std::shared_ptr<Connection> conn, const std::string& error) {
            on_connection_error(conn, error);
        });
    
    stats_.start_time = std::chrono::steady_clock::now();
    last_timeout_check_ = stats_.start_time;
}

EnhancedEventLoop::~EnhancedEventLoop() {
    stop();
}

bool EnhancedEventLoop::bind_and_listen(int port, const std::string& host) {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("socket creation failed");
        return false;
    }
    
    // 设置socket选项
    int opt = 1;
    if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        close(listen_fd_);
        return false;
    }
    
    // 设置非阻塞
    int flags = fcntl(listen_fd_, F_GETFL, 0);
    if (flags < 0 || fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("set nonblocking failed");
        close(listen_fd_);
        return false;
    }
    
    // 绑定地址
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    if (host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            std::cerr << "Invalid host address: " << host << std::endl;
            close(listen_fd_);
            return false;
        }
    }
    
    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind failed");
        close(listen_fd_);
        return false;
    }
    
    if (listen(listen_fd_, 1024) < 0) {
        perror("listen failed");
        close(listen_fd_);
        return false;
    }
    
    // 添加监听socket到事件分发器
    auto accept_callback = [this](int fd, EventType events) {
        if (has_event(events, EventType::READ)) {
            handle_accept();
        }
    };
    
    if (!dispatcher_->add_event(listen_fd_, EventType::READ, accept_callback)) {
        std::cerr << "Failed to add listen socket to dispatcher" << std::endl;
        close(listen_fd_);
        return false;
    }
    
    host_ = host;
    port_ = port;
    
    std::cout << "Server listening on " << host_ << ": " << port_ << std::endl;
    return true;
}

void EnhancedEventLoop::run() {
    running_ = true;
    
    while (running_) {
        // 处理定时器事件
        handle_timer_events();
        
        // 等待I/O事件
        int num_events = dispatcher_->wait_events(100);  // 100ms超时
        
        if (num_events < 0) {
            if (errno == EINTR) continue;
            perror("dispatcher wait failed");
            break;
        }
        
        if (num_events > 0) {
            dispatcher_->dispatch_events();
        }
        
        // 定期检查连接超时
        auto now = std::chrono::steady_clock::now();
        if (now - last_timeout_check_ >= std::chrono::seconds(10)) {
            check_connection_timeouts();
            last_timeout_check_ = now;
        }
        
        // 更新统计信息
        update_statistics();
    }
}

void EnhancedEventLoop::run_in_thread() {
    if (loop_thread_) {
        return;  // 已经在运行
    }
    
    loop_thread_ = std::make_unique<std::thread>([this]() {
        run();
    });
}

void EnhancedEventLoop::stop() {
    running_ = false;
    
    if (loop_thread_ && loop_thread_->joinable()) {
        loop_thread_->join();
        loop_thread_.reset();
    }
    
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
}

void EnhancedEventLoop::add_timer(std::chrono::milliseconds delay, TimerCallback callback) {
    auto expire_time = std::chrono::steady_clock::now() + delay;
    
    std::lock_guard<std::mutex> lock(timer_mutex_);
    timer_queue_.emplace(expire_time, std::move(callback));
}

void EnhancedEventLoop::add_repeating_timer(std::chrono::milliseconds interval, TimerCallback callback) {
    auto expire_time = std::chrono::steady_clock::now() + interval;
    
    std::lock_guard<std::mutex> lock(timer_mutex_);
    timer_queue_.emplace(expire_time, std::move(callback), true, interval);
}

std::shared_ptr<Connection> EnhancedEventLoop::get_connection(int fd) {
    return conn_manager_->get_connection(fd);
}

void EnhancedEventLoop::close_connection(int fd) {
    conn_manager_->remove_connection(fd);
}

void EnhancedEventLoop::broadcast_message(const std::string& message) {
    conn_manager_->broadcast(message);
}

size_t EnhancedEventLoop::get_connection_count() const {
    return conn_manager_->get_connection_count();
}

void EnhancedEventLoop::set_keepalive_timeout(std::chrono::seconds timeout) {
    keepalive_timeout_ = timeout;
    conn_manager_->set_keepalive_timeout(timeout);
}

void EnhancedEventLoop::set_max_connections(size_t max_conn) {
    // 这里需要重新创建连接管理器，或者添加动态调整功能
}

void EnhancedEventLoop::reset_statistics() {
    stats_.total_connections = 0;
    stats_.active_connections = 0;
    stats_.total_requests = 0;
    stats_.bytes_read = 0;
    stats_.bytes_written = 0;
    stats_.start_time = std::chrono::steady_clock::now();
}

void EnhancedEventLoop::handle_accept() {
    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(listen_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;  // 所有连接都已处理
            }
            perror("accept failed");
            break;
        }
        
        // 设置非阻塞
        int flags = fcntl(client_fd, F_GETFL, 0);
        if (flags < 0 || fcntl(client_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            perror("set client nonblocking failed");
            close(client_fd);
            continue;
        }
        
        // 检查连接数限制
        if (conn_manager_->is_full()) {
            std::cerr << "Connection limit reached, rejecting new connection" << std::endl;
            close(client_fd);
            continue;
        }
        
        // 添加到连接管理器
        auto conn = conn_manager_->add_connection(client_fd, client_addr);
        if (!conn) {
            close(client_fd);
            continue;
        }
        
        // 添加到事件分发器
        auto client_callback = [this](int fd, EventType events) {
            if (has_event(events, EventType::READ)) {
                conn_manager_->handle_read_event(fd);
            }
            if (has_event(events, EventType::WRITE)) {
                conn_manager_->handle_write_event(fd);
            }
            if (has_event(events, EventType::ERROR | EventType::HANGUP)) {
                conn_manager_->handle_error_event(fd);
            }
        };
        
        EventType events = EventType::READ;
        if (dispatcher_->supports_edge_trigger()) {
            events = events | EventType::EDGE_TRIGGERED;
        }
        
        if (!dispatcher_->add_event(client_fd, events, client_callback)) {
            std::cerr << "Failed to add client to dispatcher" << std::endl;
            conn_manager_->remove_connection(client_fd);
            continue;
        }
        
        stats_.total_connections++;
        stats_.active_connections++;
        
        // 调用用户回调
        if (accept_cb_) {
            accept_cb_(client_fd, client_addr);
        }
    }
}

void EnhancedEventLoop::handle_timer_events() {
    std::lock_guard<std::mutex> lock(timer_mutex_);
    auto now = std::chrono::steady_clock::now();
    
    while (!timer_queue_.empty() && timer_queue_.top().expire_time <= now) {
        auto task = timer_queue_.top();
        timer_queue_.pop();
        
        // 执行回调
        if (task.callback) {
            task.callback();
        }
        
        // 如果是重复定时器，重新加入队列
        if (task.repeating) {
            task.expire_time = now + task.interval;
            timer_queue_.push(task);
        }
    }
}

void EnhancedEventLoop::check_connection_timeouts() {
    conn_manager_->check_timeouts(keepalive_timeout_);
}

void EnhancedEventLoop::update_statistics() {
    stats_.active_connections = conn_manager_->get_connection_count();
}

void EnhancedEventLoop::on_new_connection(std::shared_ptr<Connection> conn) {
    // 连接建立时的处理
}

void EnhancedEventLoop::on_message_received(std::shared_ptr<Connection> conn, const char* data, size_t size) {
    stats_.bytes_read += size;
    stats_.total_requests++;
    
    if (message_cb_) {
        message_cb_(conn, data, size);
    }
}

void EnhancedEventLoop::on_connection_closed(std::shared_ptr<Connection> conn) {
    dispatcher_->remove_event(conn->get_fd());
    stats_.active_connections--;
    
    if (close_cb_) {
        close_cb_(conn);
    }
}

void EnhancedEventLoop::on_connection_error(std::shared_ptr<Connection> conn, const std::string& error) {
    if (error_cb_) {
        error_cb_(conn, error);
    }
}

// EventLoopBuilder 实现
EventLoopBuilder::EventLoopBuilder()
    : dispatcher_type_(EventDispatcherFactory::Type::AUTO),
      max_connections_(10000),
      keepalive_timeout_(std::chrono::seconds(300)) {
}

EventLoopBuilder& EventLoopBuilder::with_dispatcher(EventDispatcherFactory::Type type) {
    dispatcher_type_ = type;
    return *this;
}

EventLoopBuilder& EventLoopBuilder::with_max_connections(size_t max_conn) {
    max_connections_ = max_conn;
    return *this;
}

EventLoopBuilder& EventLoopBuilder::with_keepalive_timeout(std::chrono::seconds timeout) {
    keepalive_timeout_ = timeout;
    return *this;
}

EventLoopBuilder& EventLoopBuilder::with_accept_callback(EnhancedEventLoop::AcceptCallback cb) {
    accept_cb_ = std::move(cb);
    return *this;
}

EventLoopBuilder& EventLoopBuilder::with_message_callback(EnhancedEventLoop::MessageCallback cb) {
    message_cb_ = std::move(cb);
    return *this;
}

EventLoopBuilder& EventLoopBuilder::with_close_callback(EnhancedEventLoop::CloseCallback cb) {
    close_cb_ = std::move(cb);
    return *this;
}

EventLoopBuilder& EventLoopBuilder::with_error_callback(EnhancedEventLoop::ErrorCallback cb) {
    error_cb_ = std::move(cb);
    return *this;
}

std::unique_ptr<EnhancedEventLoop> EventLoopBuilder::build() {
    auto loop = std::make_unique<EnhancedEventLoop>(dispatcher_type_, max_connections_);
    
    loop->set_keepalive_timeout(keepalive_timeout_);
    
    if (accept_cb_) loop->set_accept_callback(accept_cb_);
    if (message_cb_) loop->set_message_callback(message_cb_);
    if (close_cb_) loop->set_close_callback(close_cb_);
    if (error_cb_) loop->set_error_callback(error_cb_);
    
    return loop;
}

// MultiThreadEventLoop 实现
MultiThreadEventLoop::MultiThreadEventLoop(size_t num_threads)
    : next_loop_(0), running_(false), listen_fd_(-1), port_(0) {
    
    for (size_t i = 0; i < num_threads; ++i) {
        event_loops_.push_back(std::make_unique<EnhancedEventLoop>());
    }
}

MultiThreadEventLoop::~MultiThreadEventLoop() {
    stop();
}

bool MultiThreadEventLoop::bind_and_listen(int port, const std::string& host) {
    // 只在主事件循环中监听
    if (!event_loops_.empty()) {
        host_ = host;
        port_ = port;
        return event_loops_[0]->bind_and_listen(port, host);
    }
    return false;
}

void MultiThreadEventLoop::run() {
    running_ = true;
    
    // 启动所有工作线程
    for (size_t i = 1; i < event_loops_.size(); ++i) {
        threads_.push_back(std::make_unique<std::thread>([this, i]() {
            event_loops_[i]->run();
        }));
    }
    
    // 主线程运行第一个事件循环
    if (!event_loops_.empty()) {
        event_loops_[0]->run();
    }
}

void MultiThreadEventLoop::stop() {
    running_ = false;
    
    for (auto& loop : event_loops_) {
        loop->stop();
    }
    
    for (auto& thread : threads_) {
        if (thread->joinable()) {
            thread->join();
        }
    }
    
    threads_.clear();
}

void MultiThreadEventLoop::set_message_callback(EnhancedEventLoop::MessageCallback cb) {
    message_cb_ = cb;
    for (auto& loop : event_loops_) {
        loop->set_message_callback(cb);
    }
}

void MultiThreadEventLoop::set_close_callback(EnhancedEventLoop::CloseCallback cb) {
    close_cb_ = cb;
    for (auto& loop : event_loops_) {
        loop->set_close_callback(cb);
    }
}

void MultiThreadEventLoop::set_error_callback(EnhancedEventLoop::ErrorCallback cb) {
    error_cb_ = cb;
    for (auto& loop : event_loops_) {
        loop->set_error_callback(cb);
    }
}

EnhancedEventLoop::Statistics MultiThreadEventLoop::get_total_statistics() const {
    EnhancedEventLoop::Statistics total;
    
    for (const auto& loop : event_loops_) {
        const auto& stats = loop->get_statistics();
        total.total_connections += stats.total_connections.load();
        total.active_connections += stats.active_connections.load();
        total.total_requests += stats.total_requests.load();
        total.bytes_read += stats.bytes_read.load();
        total.bytes_written += stats.bytes_written.load();
    }
    
    return total;
}

EnhancedEventLoop* MultiThreadEventLoop::get_next_loop() {
    if (event_loops_.empty()) return nullptr;
    
    size_t index = next_loop_.fetch_add(1) % event_loops_.size();
    return event_loops_[index].get();
}