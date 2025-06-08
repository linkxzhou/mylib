#ifndef REACTOR_SERVER_H
#define REACTOR_SERVER_H

#include "./base/server_base.h"
#include "./base/event_dispatcher.h"
#include "./base/poll_dispatcher.h"
#include <map>
#include <functional>
#include <memory>

// 使用base/event_dispatcher.h中定义的EventHandler

// 客户端连接处理器
class ClientHandler : public EventHandler {
public:
    ClientHandler(int client_fd) : client_fd_(client_fd) {}
    
    ~ClientHandler() {
        if (client_fd_ != -1) {
            close(client_fd_);
        }
    }
    
    void handle_read(int fd) override {
        char buffer[1024] = {0};
        ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read <= 0) {
            if (bytes_read == 0) {
                std::cout << "Client disconnected (fd: " << client_fd_ << ")" << std::endl;
            } else {
                perror("read failed");
            }
            return;
        }
        
        std::cout << "Received data from client (fd: " << fd << "): " << buffer << std::endl;
        
        // 准备响应数据
        response_data_ = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: 13\r\n"
            "\r\n"
            "Hello, World!";
        
        // 发送响应
        handle_write(fd);
    }
    
    void handle_write(int fd) override {
        if (!response_data_.empty()) {
            ssize_t bytes_sent = write(fd, response_data_.c_str(), response_data_.length());
            if (bytes_sent < 0) {
                perror("write failed");
            } else {
                response_data_.clear();
            }
        }
    }
    
    void handle_error(int fd) override {
        std::cerr << "Error on client fd: " << fd << std::endl;
    }
    
    void handle_hangup(int fd) override {
        std::cerr << "Client hangup on fd: " << fd << std::endl;
    }
    
    int get_fd() const {
        return client_fd_;
    }
    
private:
    int client_fd_;
    std::string response_data_;
};

// Reactor模式服务器
class ReactorServer : public ServerBase {
public:
    ReactorServer(int port = 8080) : ServerBase(port), dispatcher_(std::unique_ptr<EventDispatcher>(new PollDispatcher(1024))) {}
    
    ~ReactorServer() {
        // 清理所有事件处理器
        for (auto& pair : handlers_) {
            delete pair.second;
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
        
        running_ = true;
        std::cout << "Reactor Server started on port " << port_ << std::endl;
        
        // 事件循环
        event_loop();
        
        return true;
    }
    
    void stop() override {
        running_ = false;
        // 清理所有事件处理器
        for (auto& pair : handlers_) {
            delete pair.second;
        }
        handlers_.clear();
        ServerBase::stop();
    }
    
private:
    void event_loop() {
        // 添加监听socket到事件分发器
        dispatcher_->add_event(server_fd_, EventType::READ | EventType::ERROR, 
            [this](int fd, EventType events) {
                if (has_event(events, EventType::ERROR)) {
                    std::cerr << "Error on server socket" << std::endl;
                    running_ = false;
                    return;
                }
                if (has_event(events, EventType::READ)) {
                    handle_accept();
                }
            });
        
        while (running_) {
            // 等待事件，超时时间1秒
            int activity = dispatcher_->wait_events(1000);
            
            if (activity < 0) {
                if (errno == EINTR) continue;
                perror("poll failed");
                break;
            }
            
            if (activity == 0) {
                // 超时，继续循环
                continue;
            }
            
            // 分发事件
            dispatcher_->dispatch_events();
        }
    }
    
    void handle_accept() {
        while (true) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // 所有连接都已处理完
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
            
            // 创建事件处理器
            ClientHandler* handler = new ClientHandler(client_fd);
            handlers_[client_fd] = handler;
            
            // 添加客户端socket到事件分发器
            dispatcher_->add_event(client_fd, EventType::READ | EventType::ERROR | EventType::HANGUP,
                [this, handler](int fd, EventType events) {
                    bool should_remove = false;
                    
                    if (has_event(events, EventType::ERROR)) {
                        handler->handle_error(fd);
                        should_remove = true;
                    } else if (has_event(events, EventType::HANGUP)) {
                        handler->handle_hangup(fd);
                        should_remove = true;
                    } else if (has_event(events, EventType::READ)) {
                        handler->handle_read(fd);
                        // HTTP/1.0风格，读取后立即关闭
                        should_remove = true;
                    }
                    
                    if (should_remove) {
                        remove_handler(fd);
                    }
                });
            
            #if DEBUG
            std::cout << "New client connected: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) << " (fd: " << client_fd << ")" << std::endl;
            #endif
        }
    }
    
    void remove_handler(int fd) {
        auto it = handlers_.find(fd);
        if (it != handlers_.end()) {
            dispatcher_->remove_event(fd);
            delete it->second;
            handlers_.erase(it);
        }
    }
    
private:
    std::map<int, EventHandler*> handlers_;
    std::unique_ptr<EventDispatcher> dispatcher_;
};

#endif // REACTOR_SERVER_H