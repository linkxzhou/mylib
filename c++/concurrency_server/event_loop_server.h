#ifndef EVENT_LOOP_SERVER_H
#define EVENT_LOOP_SERVER_H

#include "./base/server_base.h"
#include "./base/kqueue_dispatcher.h"
#include <functional>
#include <queue>
#include <map>
#include <memory>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <algorithm>

// 事件循环模型 - 单线程事件循环，类似Node.js
class EventLoopServer : public ServerBase {
public:
    using Callback = std::function<void()>;
    
    struct Event {
        int fd;
        bool read_event;
        bool write_event;
        Callback callback;
        
        Event() : fd(-1), read_event(false), write_event(false) {}
        Event(int f, bool r, bool w, Callback cb) : fd(f), read_event(r), write_event(w), callback(cb) {}
    };
    
private:
    std::unique_ptr<KqueueDispatcher> dispatcher_;
    std::queue<Callback> pending_callbacks_;
    bool running_;
    
public:
    EventLoopServer(int port = 8080) : ServerBase(port), running_(false) {
        dispatcher_.reset(new KqueueDispatcher(1024));
    }
    
    ~EventLoopServer() {}
    
    void add_event(int fd, bool read_event, bool write_event, Callback callback) {
        EventType events = static_cast<EventType>(0);
        if (read_event) {
            events = events | EventType::READ;
        }
        if (write_event) {
            events = events | EventType::WRITE;
        }
        
        dispatcher_->add_event(fd, events, [callback](int fd, EventType events) {
            callback();
        });
    }
    
    void remove_event(int fd) {
        dispatcher_->remove_event(fd);
    }
    
    void next_tick(Callback callback) {
        pending_callbacks_.push(callback);
    }
    
    void process_pending_callbacks() {
        while (!pending_callbacks_.empty()) {
            auto callback = pending_callbacks_.front();
            pending_callbacks_.pop();
            callback();
        }
    }
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 设置监听socket为非阻塞
        if (!set_nonblocking(server_fd_)) {
            perror("set_nonblocking failed");
            return false;
        }
        
        std::cout << "Event Loop Server listening on port " << port_ << std::endl;
        
        // 添加监听socket到事件循环
        add_event(server_fd_, true, false, [this]() {
            handle_accept();
        });
        
        running_ = true;
        
        // 主事件循环
        while (running_) {
            // 等待事件，超时时间1秒
            int num_events = dispatcher_->wait_events(1000);
            
            if (num_events < 0) {
                if (errno == EINTR) continue;
                perror("kqueue wait failed");
                break;
            }
            
            if (num_events == 0) {
                // 超时，处理待处理的回调
                process_pending_callbacks();
                continue;
            }
            
            // 分发就绪的事件
            dispatcher_->dispatch_events();
            
            // 处理待处理的回调
            process_pending_callbacks();
        }
        
        return true;
    }
    
    void stop() override {
        running_ = false;
    }
    
private:
    void handle_accept() {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                perror("accept failed");
            }
            return;
        }
        
        // 设置客户端socket为非阻塞
        if (!set_nonblocking(client_fd)) {
            close(client_fd);
            return;
        }
        
        #ifdef DEBUG
        std::cout << "New client connected: " << client_fd << std::endl;
        #endif
        
        // 添加客户端socket到事件循环
        add_event(client_fd, true, false, [this, client_fd]() {
            handle_client(client_fd);
        });
    }
    
    void handle_client(int client_fd) override {
        char buffer[1024];
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read <= 0) {
            if (bytes_read == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
                std::cout << "Client disconnected: " << client_fd << std::endl;
                remove_event(client_fd);
                close(client_fd);
            }
            return;
        }
        
        buffer[bytes_read] = '\0';
        std::cout << "Received from client " << client_fd << ": " << buffer;
        
        // 异步处理请求（模拟）
        std::string data = std::string(buffer);
        next_tick([this, client_fd, data]() {
            process_request_async(client_fd, data);
        });
    }
    
    void process_request_async(int client_fd, const std::string& data) {
        // 模拟异步处理
        std::string response = "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/plain\r\n"
                              "Content-Length: 13\r\n"
                              "\r\n"
                              "Hello, World!";
        
        // 发送响应
        ssize_t bytes_sent = write(client_fd, response.c_str(), response.length());
        if (bytes_sent < 0) {
            perror("write failed");
        }
        
        // 关闭连接
        remove_event(client_fd);
        close(client_fd);
    }
};

#endif // EVENT_LOOP_SERVER_H