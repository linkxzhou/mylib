#ifndef HYBRID_SERVER_H
#define HYBRID_SERVER_H

#include "./base/server_base.h"
#include "./base/event_dispatcher.h"
#include "./base/kqueue_dispatcher.h"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <unordered_map>

// 使用base/event_dispatcher.h中定义的EventType

// 连接上下文
struct Connection {
    int fd;
    struct sockaddr_in addr;
    std::string read_buffer;
    std::string write_buffer;
    bool write_pending;
    
    Connection(int client_fd, const struct sockaddr_in& client_addr)
        : fd(client_fd), addr(client_addr), write_pending(false) {}
};

// 任务类型
struct Task {
    std::shared_ptr<Connection> conn;
    EventType event_type;
    
    Task(std::shared_ptr<Connection> c, EventType type)
        : conn(c), event_type(type) {}
};

// 使用server_base.h中定义的ThreadPool

// Reactor事件循环
class ReactorEventLoop {
public:
    ReactorEventLoop(ThreadPool& thread_pool) : thread_pool_(thread_pool), running_(true) {
        dispatcher_.reset(new KqueueDispatcher(1024));
    }
    
    ~ReactorEventLoop() {
        // KqueueDispatcher 的析构函数会自动清理资源
    }
    
    bool add_connection(std::shared_ptr<Connection> conn) {
        connections_[conn->fd] = conn;
        
        // 使用 KqueueDispatcher 添加读事件
        auto callback = [this](int fd, EventType events) {
            this->handle_event(fd, events);
        };
        
        return dispatcher_->add_event(conn->fd, EventType::READ, callback);
    }
    
    void remove_connection(int fd) {
        auto it = connections_.find(fd);
        if (it != connections_.end()) {
            dispatcher_->remove_event(fd);
            close(fd);
            connections_.erase(it);
        }
    }
    
    void run() {
        running_ = true;
        std::cout << "Reactor event loop started" << std::endl;
        
        while (running_) {
            // 使用 KqueueDispatcher 等待事件
            int num_events = dispatcher_->wait_events(1000);  // 1秒超时
            
            if (num_events > 0) {
                // 分发事件
                dispatcher_->dispatch_events();
            } else if (num_events == 0) {
                // 超时，继续循环
                continue;
            } else {
                // 错误处理
                if (errno != EINTR) {
                    perror("wait_events failed");
                    break;
                }
            }
        }
        
        std::cout << "Reactor event loop stopped" << std::endl;
    }
    
    void stop() {
        running_ = false;
    }
    
private:
    void handle_event(int fd, EventType events) {
        auto it = connections_.find(fd);
        if (it == connections_.end()) {
            return;
        }
        
        if (has_event(events, EventType::READ)) {
            // 读事件，提交给线程池处理
            Task task(it->second, EventType::READ);
            thread_pool_.enqueue([task, this]() {
                // 处理读事件的逻辑
                char buffer[1024] = {0};
                ssize_t bytes_read = read(task.conn->fd, buffer, sizeof(buffer) - 1);
                
                if (bytes_read > 0) {
                    std::cout << "Read " << bytes_read << " bytes from fd " << task.conn->fd << std::endl;
                    
                    // 生成响应
                    std::string response = 
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/plain\r\n"
                        "Content-Length: 13\r\n"
                        "\r\n"
                        "Hello, World!";
                    
                    // 发送响应
                    write(task.conn->fd, response.c_str(), response.length());
                    this->remove_connection(task.conn->fd);
                } else if (bytes_read == 0) {
                    // 连接关闭
                    this->remove_connection(task.conn->fd);
                }
            });
        }
        
        if (has_event(events, EventType::HANGUP) || has_event(events, EventType::ERROR)) {
            // 连接错误或关闭
            remove_connection(fd);
        }
    }
    
    ThreadPool& thread_pool_;
    std::atomic<bool> running_;
    std::unordered_map<int, std::shared_ptr<Connection>> connections_;
    std::unique_ptr<KqueueDispatcher> dispatcher_;
};

// 混合模型服务器 (Reactor + 线程池)
class HybridServer : public ServerBase {
public:
    HybridServer(int port = 8080, int num_threads = 4)
        : ServerBase(port), thread_pool_(num_threads), reactor_(thread_pool_) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 启动线程池
        // ThreadPool在构造时自动启动，无需调用start方法
        
        // 启动Reactor事件循环
        reactor_thread_ = std::thread(&ReactorEventLoop::run, &reactor_);
        
        running_ = true;
        std::cout << "Hybrid Server (Reactor + ThreadPool) started on port " << port_ << std::endl;
        
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                perror("accept failed");
                break;
            }
            
            #ifdef DEBUG
            std::cout << "Client connected: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) << std::endl;
            #endif
            
            // 设置非阻塞模式
            int flags = fcntl(client_fd, F_GETFL, 0);
            fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
            
            // 创建连接对象并添加到Reactor
            auto conn = std::make_shared<Connection>(client_fd, client_addr);
            reactor_.add_connection(conn);
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        reactor_.stop();
        
        if (reactor_thread_.joinable()) {
            reactor_thread_.join();
        }
        
        // ThreadPool的析构函数会自动停止线程
    }
    
private:
    ThreadPool thread_pool_;
    ReactorEventLoop reactor_;
    std::thread reactor_thread_;
};

#endif // HYBRID_SERVER_H