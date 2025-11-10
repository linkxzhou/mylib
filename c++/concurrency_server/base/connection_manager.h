#ifndef CONNECTION_MANAGER_H
#define CONNECTION_MANAGER_H

#include "server_base.h"
#include "event_dispatcher.h"
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>

// 连接状态枚举
enum class ConnectionState {
    CONNECTING,
    CONNECTED,
    READING,
    WRITING,
    CLOSING,
    CLOSED
};

// 连接统计信息
struct ConnectionStats {
    std::chrono::steady_clock::time_point connect_time;
    std::chrono::steady_clock::time_point last_activity;
    size_t bytes_read = 0;
    size_t bytes_written = 0;
    size_t requests_handled = 0;
};

// 连接类 - 封装单个客户端连接
class Connection {
public:
    Connection(int fd, const struct sockaddr_in& addr);
    ~Connection();
    
    // 基本属性
    int get_fd() const { return fd_; }
    ConnectionState get_state() const { return state_; }
    void set_state(ConnectionState state) { state_ = state; update_activity(); }
    
    // 地址信息
    std::string get_remote_ip() const;
    int get_remote_port() const;
    
    // 数据读写
    ssize_t read_data(char* buffer, size_t size);
    ssize_t write_data(const char* data, size_t size);
    
    // 缓冲区管理
    void append_to_write_buffer(const std::string& data);
    bool has_pending_writes() const { return !write_buffer_.empty(); }
    bool flush_write_buffer();
    
    // 超时检查
    bool is_timeout(std::chrono::seconds timeout) const;
    void update_activity() { stats_.last_activity = std::chrono::steady_clock::now(); }
    
    // 统计信息
    const ConnectionStats& get_stats() const { return stats_; }
    
    // 关闭连接
    void close();
    bool is_closed() const { return state_ == ConnectionState::CLOSED; }
    
private:
    int fd_;
    struct sockaddr_in addr_;
    ConnectionState state_;
    ConnectionStats stats_;
    std::string write_buffer_;
    std::atomic<bool> closed_;
};

// 连接管理器 - 管理所有客户端连接
class ConnectionManager {
public:
    using ConnectionPtr = std::shared_ptr<Connection>;
    using NewConnectionCallback = std::function<void(ConnectionPtr)>;
    using DataReceivedCallback = std::function<void(ConnectionPtr, const char*, size_t)>;
    using ConnectionClosedCallback = std::function<void(ConnectionPtr)>;
    using ErrorCallback = std::function<void(ConnectionPtr, const std::string&)>;
    
    explicit ConnectionManager(size_t max_connections = 10000);
    ~ConnectionManager();
    
    // 连接管理
    ConnectionPtr add_connection(int fd, const struct sockaddr_in& addr);
    void remove_connection(int fd);
    ConnectionPtr get_connection(int fd);
    
    // 回调设置
    void set_new_connection_callback(NewConnectionCallback cb) { new_connection_cb_ = std::move(cb); }
    void set_data_received_callback(DataReceivedCallback cb) { data_received_cb_ = std::move(cb); }
    void set_connection_closed_callback(ConnectionClosedCallback cb) { connection_closed_cb_ = std::move(cb); }
    void set_error_callback(ErrorCallback cb) { error_cb_ = std::move(cb); }
    
    // 事件处理
    void handle_read_event(int fd);
    void handle_write_event(int fd);
    void handle_error_event(int fd);
    void handle_close_event(int fd);
    
    // 超时管理
    void check_timeouts(std::chrono::seconds timeout);
    void set_keepalive_timeout(std::chrono::seconds timeout) { keepalive_timeout_ = timeout; }
    
    // 统计信息
    size_t get_connection_count() const { return connections_.size(); }
    size_t get_max_connections() const { return max_connections_; }
    bool is_full() const { return connections_.size() >= max_connections_; }
    
    // 广播消息
    void broadcast(const std::string& message);
    
    // 清理所有连接
    void close_all_connections();
    
private:
    std::unordered_map<int, ConnectionPtr> connections_;
    size_t max_connections_;
    std::chrono::seconds keepalive_timeout_;
    
    // 回调函数
    NewConnectionCallback new_connection_cb_;
    DataReceivedCallback data_received_cb_;
    ConnectionClosedCallback connection_closed_cb_;
    ErrorCallback error_cb_;
    
    // 内部方法
    void notify_new_connection(ConnectionPtr conn);
    void notify_data_received(ConnectionPtr conn, const char* data, size_t size);
    void notify_connection_closed(ConnectionPtr conn);
    void notify_error(ConnectionPtr conn, const std::string& error);
};

// 连接池 - 复用连接对象
class ConnectionPool {
public:
    explicit ConnectionPool(size_t initial_size = 100);
    ~ConnectionPool();
    
    std::shared_ptr<Connection> acquire(int fd, const struct sockaddr_in& addr);
    void release(std::shared_ptr<Connection> conn);
    
    size_t size() const { return pool_.size(); }
    
private:
    std::vector<std::shared_ptr<Connection>> pool_;
    std::mutex mutex_;
};

#endif // CONNECTION_MANAGER_H