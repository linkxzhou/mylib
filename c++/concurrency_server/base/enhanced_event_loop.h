#ifndef ENHANCED_EVENT_LOOP_H
#define ENHANCED_EVENT_LOOP_H

#include "event_dispatcher.h"
#include "connection_manager.h"
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>

// 定时器任务
struct TimerTask {
    std::chrono::steady_clock::time_point expire_time;
    std::function<void()> callback;
    bool repeating;
    std::chrono::milliseconds interval;
    
    TimerTask(std::chrono::steady_clock::time_point exp, std::function<void()> cb, 
              bool repeat = false, std::chrono::milliseconds intv = std::chrono::milliseconds(0))
        : expire_time(exp), callback(std::move(cb)), repeating(repeat), interval(intv) {}
    
    bool operator>(const TimerTask& other) const {
        return expire_time > other.expire_time;
    }
};

// 增强的事件循环 - 集成连接管理和定时器
class EnhancedEventLoop {
public:
    using AcceptCallback = std::function<void(int client_fd, const struct sockaddr_in& addr)>;
    using MessageCallback = std::function<void(std::shared_ptr<Connection>, const char*, size_t)>;
    using CloseCallback = std::function<void(std::shared_ptr<Connection>)>;
    using ErrorCallback = std::function<void(std::shared_ptr<Connection>, const std::string&)>;
    using TimerCallback = std::function<void()>;
    
    explicit EnhancedEventLoop(EventDispatcherFactory::Type dispatcher_type = EventDispatcherFactory::Type::AUTO,
                              size_t max_connections = 10000);
    ~EnhancedEventLoop();
    
    // 服务器设置
    bool bind_and_listen(int port, const std::string& host = "0.0.0.0");
    
    // 事件循环控制
    void run();
    void run_in_thread();
    void stop();
    bool is_running() const { return running_; }
    
    // 回调设置
    void set_accept_callback(AcceptCallback cb) { accept_cb_ = std::move(cb); }
    void set_message_callback(MessageCallback cb) { message_cb_ = std::move(cb); }
    void set_close_callback(CloseCallback cb) { close_cb_ = std::move(cb); }
    void set_error_callback(ErrorCallback cb) { error_cb_ = std::move(cb); }
    
    // 定时器管理
    void add_timer(std::chrono::milliseconds delay, TimerCallback callback);
    void add_repeating_timer(std::chrono::milliseconds interval, TimerCallback callback);
    
    // 连接管理
    std::shared_ptr<Connection> get_connection(int fd);
    void close_connection(int fd);
    void broadcast_message(const std::string& message);
    size_t get_connection_count() const;
    
    // 配置
    void set_keepalive_timeout(std::chrono::seconds timeout);
    void set_max_connections(size_t max_conn);
    
    // 统计信息
    struct Statistics {
        std::atomic<uint64_t> total_connections{0};
        std::atomic<uint64_t> active_connections{0};
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> bytes_read{0};
        std::atomic<uint64_t> bytes_written{0};
        std::chrono::steady_clock::time_point start_time;
    };
    
    const Statistics& get_statistics() const { return stats_; }
    void reset_statistics();
    
private:
    // 核心组件
    std::unique_ptr<EventDispatcher> dispatcher_;
    std::unique_ptr<ConnectionManager> conn_manager_;
    
    // 服务器socket
    int listen_fd_;
    std::string host_;
    int port_;
    
    // 事件循环状态
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> loop_thread_;
    
    // 定时器
    std::priority_queue<TimerTask, std::vector<TimerTask>, std::greater<TimerTask>> timer_queue_;
    std::mutex timer_mutex_;
    
    // 回调函数
    AcceptCallback accept_cb_;
    MessageCallback message_cb_;
    CloseCallback close_cb_;
    ErrorCallback error_cb_;
    
    // 统计信息
    Statistics stats_;
    
    // 配置
    std::chrono::seconds keepalive_timeout_;
    std::chrono::steady_clock::time_point last_timeout_check_;
    
    // 内部方法
    void handle_accept();
    void handle_timer_events();
    void check_connection_timeouts();
    void update_statistics();
    
    // 事件处理
    void on_new_connection(std::shared_ptr<Connection> conn);
    void on_message_received(std::shared_ptr<Connection> conn, const char* data, size_t size);
    void on_connection_closed(std::shared_ptr<Connection> conn);
    void on_connection_error(std::shared_ptr<Connection> conn, const std::string& error);
};

// 事件循环构建器 - 简化配置
class EventLoopBuilder {
public:
    EventLoopBuilder();
    
    EventLoopBuilder& with_dispatcher(EventDispatcherFactory::Type type);
    EventLoopBuilder& with_max_connections(size_t max_conn);
    EventLoopBuilder& with_keepalive_timeout(std::chrono::seconds timeout);
    EventLoopBuilder& with_accept_callback(EnhancedEventLoop::AcceptCallback cb);
    EventLoopBuilder& with_message_callback(EnhancedEventLoop::MessageCallback cb);
    EventLoopBuilder& with_close_callback(EnhancedEventLoop::CloseCallback cb);
    EventLoopBuilder& with_error_callback(EnhancedEventLoop::ErrorCallback cb);
    
    std::unique_ptr<EnhancedEventLoop> build();
    
private:
    EventDispatcherFactory::Type dispatcher_type_;
    size_t max_connections_;
    std::chrono::seconds keepalive_timeout_;
    EnhancedEventLoop::AcceptCallback accept_cb_;
    EnhancedEventLoop::MessageCallback message_cb_;
    EnhancedEventLoop::CloseCallback close_cb_;
    EnhancedEventLoop::ErrorCallback error_cb_;
};

// 多线程事件循环 - 支持多个工作线程
class MultiThreadEventLoop {
public:
    explicit MultiThreadEventLoop(size_t num_threads = std::thread::hardware_concurrency());
    ~MultiThreadEventLoop();
    
    bool bind_and_listen(int port, const std::string& host = "0.0.0.0");
    void run();
    void stop();
    
    // 设置回调（会应用到所有线程）
    void set_message_callback(EnhancedEventLoop::MessageCallback cb);
    void set_close_callback(EnhancedEventLoop::CloseCallback cb);
    void set_error_callback(EnhancedEventLoop::ErrorCallback cb);
    
    // 获取统计信息（汇总所有线程）
    EnhancedEventLoop::Statistics get_total_statistics() const;
    
private:
    std::vector<std::unique_ptr<EnhancedEventLoop>> event_loops_;
    std::vector<std::unique_ptr<std::thread>> threads_;
    std::atomic<size_t> next_loop_;
    std::atomic<bool> running_;
    
    int listen_fd_;
    std::string host_;
    int port_;
    
    // 回调函数
    EnhancedEventLoop::MessageCallback message_cb_;
    EnhancedEventLoop::CloseCallback close_cb_;
    EnhancedEventLoop::ErrorCallback error_cb_;
    
    // 负载均衡 - 选择下一个事件循环
    EnhancedEventLoop* get_next_loop();
};

#endif // ENHANCED_EVENT_LOOP_H