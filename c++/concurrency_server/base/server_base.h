#ifndef SERVER_BASE_H
#define SERVER_BASE_H

#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>

// 服务器基类
class ServerBase {
public:
    ServerBase(int port = 8080) : port_(port), server_fd_(-1), running_(false) {}
    virtual ~ServerBase() { stop(); }
    
    virtual bool start() = 0;
    virtual void stop() {
        running_ = false;
        if (server_fd_ != -1) {
            close(server_fd_);
            server_fd_ = -1;
        }
    }
    
    bool is_running() const { return running_; }
    
protected:
    // 创建监听socket
    bool create_listen_socket() {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ == -1) {
            perror("socket creation failed");
            return false;
        }
        
        int opt = 1;
        if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            perror("setsockopt failed");
            return false;
        }
        
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port_);
        
        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            perror("bind failed");
            return false;
        }
        
        if (listen(server_fd_, 10) < 0) {
            perror("listen failed");
            return false;
        }
        
        return true;
    }
    
    // 处理客户端请求
    virtual void handle_client(int client_fd) {
        char buffer[1024] = {0};
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            // 简单的HTTP响应
            const char* response = 
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html\r\n"
                "Content-Length: 13\r\n"
                "\r\n"
                "Hello, World!";
            
            write(client_fd, response, strlen(response));
        }
        
        close(client_fd);
    }
    
    // 设置非阻塞模式
    bool set_nonblocking(int fd) {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags == -1) return false;
        return fcntl(fd, F_SETFL, flags | O_NONBLOCK) != -1;
    }
    
protected:
    int port_;
    int server_fd_;
    std::atomic<bool> running_;
};

// 线程池实现
class ThreadPool {
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        
                        if (stop_ && tasks_.empty()) return;
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) return;
            tasks_.emplace(std::forward<F>(f));
        }
        condition_.notify_one();
    }
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

#endif // SERVER_BASE_H