#ifndef COROUTINE_SERVER_H
#define COROUTINE_SERVER_H

#include "./base/server_base.h"
#include "./base/coroutine.h"
#include <memory>
#include <map>

// 使用真正的协程实现，基于base/coroutine.h

// 客户端处理协程函数
void client_coroutine_func(int client_fd, coroutine::Scheduler* scheduler) {
    std::string request_data;
    std::string response_data;
    
    // 读取请求数据
    while (true) {
        char buffer[1024] = {0};
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 需要等待更多数据，让出CPU
                scheduler->yield();
                continue;
            }
            // 读取错误，关闭连接
            close(client_fd);
            return;
        } else if (bytes_read == 0) {
            // 客户端关闭连接
            close(client_fd);
            return;
        }
        
        request_data = std::string(buffer, bytes_read);
        break;
    }
    
    std::cout << "Processing request from client (fd: " << client_fd << "): " 
              << request_data << std::endl;
    
    // 模拟异步处理，让出CPU
    scheduler->yield();
    
    // 准备响应数据
    response_data = 
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html\r\n"
        "Content-Length: 13\r\n"
        "\r\n"
        "Hello, World!";
    
    // 发送响应数据
    size_t total_sent = 0;
    while (total_sent < response_data.length()) {
        ssize_t bytes_sent = write(client_fd, 
                                   response_data.c_str() + total_sent, 
                                   response_data.length() - total_sent);
        
        if (bytes_sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 需要等待socket可写，让出CPU
                scheduler->yield();
                continue;
            }
            // 写入错误，关闭连接
            close(client_fd);
            return;
        }
        
        total_sent += bytes_sent;
        
        // 如果没有完全发送，让出CPU给其他协程
        if (total_sent < response_data.length()) {
            scheduler->yield();
        }
    }
    
    // 发送完成，关闭连接
    close(client_fd);
}

// 协程管理器
class CoroutineManager {
public:
    CoroutineManager() : scheduler_(std::unique_ptr<coroutine::Scheduler>(new coroutine::Scheduler())) {}
    
    void add_client_coroutine(int client_fd) {
        // 创建客户端处理协程
        int coro_id = scheduler_->create_coroutine([client_fd, this]() {
            client_coroutine_func(client_fd, scheduler_.get());
        });
        
        active_coroutines_[coro_id] = client_fd;
    }
    
    void run_once() {
        // 运行所有活跃的协程
        auto it = active_coroutines_.begin();
        while (it != active_coroutines_.end()) {
            int coro_id = it->first;
            
            if (scheduler_->status(coro_id) == coroutine::Status::DEAD) {
                // 协程已结束，从活跃列表中移除
                it = active_coroutines_.erase(it);
            } else {
                // 恢复协程执行
                scheduler_->resume(coro_id);
                ++it;
            }
        }
    }
    
    size_t size() const {
        return active_coroutines_.size();
    }
    
private:
    std::unique_ptr<coroutine::Scheduler> scheduler_;
    std::map<int, int> active_coroutines_;  // 协程ID -> 客户端fd的映射
};

// 协程服务器
class CoroutineServer : public ServerBase {
public:
    CoroutineServer(int port = 8080) : ServerBase(port) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 设置监听socket为非阻塞
        if (!set_nonblocking(server_fd_)) {
            perror("set_nonblocking failed");
            return false;
        }
        
        running_ = true;
        std::cout << "Coroutine Server started on port " << port_ << std::endl;
        
        while (running_) {
            // 检查新连接
            handle_new_connections();
            
            // 运行协程管理器
            coro_manager_.run_once();
            
            // 如果没有活跃的协程，短暂休眠
            if (coro_manager_.size() == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        return true;
    }
    
private:
    void handle_new_connections() {
        while (true) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // 没有新连接
                    break;
                }
                perror("accept failed");
                break;
            }
            
            // 设置客户端socket为非阻塞
            if (!set_nonblocking(client_fd)) {
                perror("set_nonblocking failed for client");
                close(client_fd);
                continue;
            }

            #ifdef DEBUG
            std::cout << "New client connected: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) << " (fd: " << client_fd << ")" << std::endl;
            #endif
            
            // 创建客户端处理协程
            coro_manager_.add_client_coroutine(client_fd);
        }
    }
    
private:
    CoroutineManager coro_manager_;
};

#endif // COROUTINE_SERVER_H