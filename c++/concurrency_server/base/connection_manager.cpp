#include "connection_manager.h"
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <algorithm>

// Connection 实现
Connection::Connection(int fd, const struct sockaddr_in& addr)
    : fd_(fd), addr_(addr), state_(ConnectionState::CONNECTING), closed_(false) {
    stats_.connect_time = std::chrono::steady_clock::now();
    stats_.last_activity = stats_.connect_time;
    state_ = ConnectionState::CONNECTED;
}

Connection::~Connection() {
    close();
}

std::string Connection::get_remote_ip() const {
    return std::string(inet_ntoa(addr_.sin_addr));
}

int Connection::get_remote_port() const {
    return ntohs(addr_.sin_port);
}

ssize_t Connection::read_data(char* buffer, size_t size) {
    if (is_closed()) return -1;
    
    ssize_t bytes_read = ::read(fd_, buffer, size);
    if (bytes_read > 0) {
        stats_.bytes_read += bytes_read;
        update_activity();
    } else if (bytes_read == 0) {
        // 连接关闭
        set_state(ConnectionState::CLOSING);
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        // 读取错误
        set_state(ConnectionState::CLOSING);
    }
    
    return bytes_read;
}

ssize_t Connection::write_data(const char* data, size_t size) {
    if (is_closed()) return -1;
    
    ssize_t bytes_written = ::write(fd_, data, size);
    if (bytes_written > 0) {
        stats_.bytes_written += bytes_written;
        update_activity();
    } else if (bytes_written < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        // 写入错误
        set_state(ConnectionState::CLOSING);
    }
    
    return bytes_written;
}

void Connection::append_to_write_buffer(const std::string& data) {
    write_buffer_.append(data);
}

bool Connection::flush_write_buffer() {
    if (write_buffer_.empty() || is_closed()) {
        return true;
    }
    
    ssize_t bytes_written = write_data(write_buffer_.c_str(), write_buffer_.size());
    if (bytes_written > 0) {
        write_buffer_.erase(0, bytes_written);
    }
    
    return write_buffer_.empty();
}

bool Connection::is_timeout(std::chrono::seconds timeout) const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - stats_.last_activity);
    return elapsed >= timeout;
}

void Connection::close() {
    if (!closed_.exchange(true)) {
        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }
        state_ = ConnectionState::CLOSED;
    }
}

// ConnectionManager 实现
ConnectionManager::ConnectionManager(size_t max_connections)
    : max_connections_(max_connections), keepalive_timeout_(std::chrono::seconds(300)) {
}

ConnectionManager::~ConnectionManager() {
    close_all_connections();
}

ConnectionManager::ConnectionPtr ConnectionManager::add_connection(int fd, const struct sockaddr_in& addr) {
    if (is_full()) {
        return nullptr;
    }
    
    auto conn = std::make_shared<Connection>(fd, addr);
    connections_[fd] = conn;
    
    notify_new_connection(conn);
    return conn;
}

void ConnectionManager::remove_connection(int fd) {
    auto it = connections_.find(fd);
    if (it != connections_.end()) {
        auto conn = it->second;
        connections_.erase(it);
        
        if (!conn->is_closed()) {
            conn->close();
        }
        
        notify_connection_closed(conn);
    }
}

ConnectionManager::ConnectionPtr ConnectionManager::get_connection(int fd) {
    auto it = connections_.find(fd);
    return (it != connections_.end()) ? it->second : nullptr;
}

void ConnectionManager::handle_read_event(int fd) {
    auto conn = get_connection(fd);
    if (!conn) return;
    
    char buffer[4096];
    ssize_t bytes_read = conn->read_data(buffer, sizeof(buffer) - 1);
    
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        conn->set_state(ConnectionState::READING);
        notify_data_received(conn, buffer, bytes_read);
    } else if (bytes_read == 0) {
        // 连接关闭
        handle_close_event(fd);
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        // 读取错误
        handle_error_event(fd);
    }
}

void ConnectionManager::handle_write_event(int fd) {
    auto conn = get_connection(fd);
    if (!conn) return;
    
    conn->set_state(ConnectionState::WRITING);
    
    if (conn->has_pending_writes()) {
        if (conn->flush_write_buffer()) {
            conn->set_state(ConnectionState::CONNECTED);
        }
    }
}

void ConnectionManager::handle_error_event(int fd) {
    auto conn = get_connection(fd);
    if (!conn) return;
    
    notify_error(conn, "Socket error occurred");
    remove_connection(fd);
}

void ConnectionManager::handle_close_event(int fd) {
    remove_connection(fd);
}

void ConnectionManager::check_timeouts(std::chrono::seconds timeout) {
    std::vector<int> timeout_fds;
    
    for (const auto& pair : connections_) {
        if (pair.second->is_timeout(timeout)) {
            timeout_fds.push_back(pair.first);
        }
    }
    
    for (int fd : timeout_fds) {
        auto conn = get_connection(fd);
        if (conn) {
            notify_error(conn, "Connection timeout");
        }
        remove_connection(fd);
    }
}

void ConnectionManager::broadcast(const std::string& message) {
    for (const auto& pair : connections_) {
        auto conn = pair.second;
        if (!conn->is_closed()) {
            conn->append_to_write_buffer(message);
        }
    }
}

void ConnectionManager::close_all_connections() {
    std::vector<int> all_fds;
    for (const auto& pair : connections_) {
        all_fds.push_back(pair.first);
    }
    
    for (int fd : all_fds) {
        remove_connection(fd);
    }
}

void ConnectionManager::notify_new_connection(ConnectionPtr conn) {
    if (new_connection_cb_) {
        new_connection_cb_(conn);
    }
}

void ConnectionManager::notify_data_received(ConnectionPtr conn, const char* data, size_t size) {
    if (data_received_cb_) {
        data_received_cb_(conn, data, size);
    }
}

void ConnectionManager::notify_connection_closed(ConnectionPtr conn) {
    if (connection_closed_cb_) {
        connection_closed_cb_(conn);
    }
}

void ConnectionManager::notify_error(ConnectionPtr conn, const std::string& error) {
    if (error_cb_) {
        error_cb_(conn, error);
    }
}

// ConnectionPool 实现
ConnectionPool::ConnectionPool(size_t initial_size) {
    pool_.reserve(initial_size);
}

ConnectionPool::~ConnectionPool() {
    pool_.clear();
}

std::shared_ptr<Connection> ConnectionPool::acquire(int fd, const struct sockaddr_in& addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!pool_.empty()) {
        auto conn = pool_.back();
        pool_.pop_back();
        // 重新初始化连接对象
        return std::make_shared<Connection>(fd, addr);
    }
    
    return std::make_shared<Connection>(fd, addr);
}

void ConnectionPool::release(std::shared_ptr<Connection> conn) {
    if (!conn || conn->is_closed()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 限制池大小
    if (pool_.size() < 1000) {
        pool_.push_back(conn);
    }
}