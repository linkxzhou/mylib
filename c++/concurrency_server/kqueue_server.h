#ifndef KQUEUE_SERVER_H
#define KQUEUE_SERVER_H

#include "./base/server_base.h"
#include "./base/event_dispatcher.h"
#include "./base/kqueue_dispatcher.h"

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/event.h>

// Kqueue服务器 - 使用kqueue进行高效I/O多路复用（BSD/macOS特有）
class KqueueServer : public ServerBase {
public:
    KqueueServer(int port = 8080, int max_events = 1024) 
        : ServerBase(port) {
        dispatcher_ = std::unique_ptr<EventDispatcher>(new KqueueDispatcher(max_events));
    }
    
    ~KqueueServer() = default;
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 设置监听socket为非阻塞
        if (!set_nonblocking(server_fd_)) {
            perror("set_nonblocking failed");
            return false;
        }
        
        // 将监听socket加入事件分发器（使用边缘触发）
        auto server_callback = [this](int fd, EventType events) {
            if (has_event(events, EventType::READ)) {
                handle_new_connection();
            }
        };
        
        if (!dispatcher_->add_event(server_fd_, EventType::READ | EventType::EDGE_TRIGGERED, server_callback)) {
            std::cerr << "Failed to add server socket to dispatcher" << std::endl;
            return false;
        }
        
        running_ = true;
        std::cout << "Kqueue Server started on port " << port_ << std::endl;
        
        // 事件循环
        while (running_) {
            int num_events = dispatcher_->wait_events(1000);  // 1秒超时
            
            if (num_events < 0) {
                if (errno == EINTR) continue;
                perror("dispatcher wait failed");
                break;
            }
            
            if (num_events > 0) {
                dispatcher_->dispatch_events();
            }
        }
        
        return true;
    }
    
    void stop() override {
        running_ = false;
        ServerBase::stop();
    }
    
private:
    void handle_new_connection() {
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
            
            // 将新客户端加入事件分发器（使用边缘触发）
            auto client_callback = [this](int fd, EventType events) {
                if (has_event(events, EventType::READ)) {
                    handle_client_data(fd);
                } else if (has_event(events, EventType::ERROR | EventType::HANGUP)) {
                    handle_client_disconnect(fd);
                }
            };
            
            if (!dispatcher_->add_event(client_fd, EventType::READ | EventType::EDGE_TRIGGERED, client_callback)) {
                std::cerr << "Failed to add client socket to dispatcher" << std::endl;
                close(client_fd);
                continue;
            }
            
            #ifdef DEBUG
            std::cout << "New client connected: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) << " (fd: " << client_fd << ")" << std::endl;
            #endif
        }
    }
    
    void handle_client_data(int client_fd) {
        char buffer[1024] = {0};
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read <= 0) {
            if (bytes_read == 0) {
                std::cout << "Client disconnected (fd: " << client_fd << ")" << std::endl;
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                perror("read failed");
            }
            
            handle_client_disconnect(client_fd);
            return;
        }
        
        #if DEBUG
        std::cout << "Received data from client (fd: " << client_fd << "): " << buffer << std::endl;
        #endif
        
        // 发送响应
        const char* response = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: 13\r\n"
            "\r\n"
            "Hello, World!";
        
        ssize_t bytes_sent = write(client_fd, response, strlen(response));
        if (bytes_sent < 0) {
            perror("write failed");
        }
        
        // 关闭连接（HTTP/1.0风格）
        handle_client_disconnect(client_fd);
    }
    
    void handle_client_disconnect(int client_fd) {
        dispatcher_->remove_event(client_fd);
        close(client_fd);
    }
    
private:
    std::unique_ptr<EventDispatcher> dispatcher_;
};

#else
// 非BSD/macOS系统的占位实现
class KqueueServer : public ServerBase {
public:
    KqueueServer(int port = 8080, int max_events = 1024) : ServerBase(port) {}
    
    bool start() override {
        std::cerr << "Kqueue is only available on BSD/macOS systems" << std::endl;
        return false;
    }
};
#endif // BSD/macOS

#endif // KQUEUE_SERVER_H