#ifndef HALF_SYNC_ASYNC_SERVER_H
#define HALF_SYNC_ASYNC_SERVER_H

#include "./base/server_base.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <memory>

// 请求数据结构
struct Request {
    int client_fd;
    std::string data;
    struct sockaddr_in client_addr;
    
    Request(int fd, const std::string& d, const struct sockaddr_in& addr)
        : client_fd(fd), data(d), client_addr(addr) {}
};

// 异步响应数据结构
struct AsyncResponse {
    int client_fd;
    std::string data;
    
    AsyncResponse(int fd, const std::string& response_data)
        : client_fd(fd), data(response_data) {}
};

// 异步I/O层
class AsyncIOLayer {
public:
    AsyncIOLayer() : running_(false) {}
    
    void start() {
        running_ = true;
        io_thread_ = std::thread(&AsyncIOLayer::io_loop, this);
    }
    
    void stop() {
        running_ = false;
        response_cv_.notify_all();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }
    
    // 异步读取客户端数据
    bool async_read(int client_fd, const struct sockaddr_in& client_addr, 
                   std::function<void(std::unique_ptr<Request>)> callback) {
        // 设置非阻塞模式
        int flags = fcntl(client_fd, F_GETFL, 0);
        fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
        
        char buffer[1024] = {0};
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            auto request = std::unique_ptr<Request>(new Request(client_fd, std::string(buffer, bytes_read), client_addr));
            callback(std::move(request));
            return true;
        } else if (bytes_read == 0) {
            // 客户端关闭连接
            close(client_fd);
            return false;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 需要稍后重试
            return false;
        } else {
            // 读取错误
            perror("async_read failed");
            close(client_fd);
            return false;
        }
    }
    
    // 异步发送响应
    void async_write(std::unique_ptr<AsyncResponse> response) {
        std::lock_guard<std::mutex> lock(response_mutex_);
        response_queue_.push(std::move(response));
        response_cv_.notify_one();
    }
    
private:
    void io_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(response_mutex_);
            response_cv_.wait(lock, [this] { 
                return !response_queue_.empty() || !running_; 
            });
            
            while (!response_queue_.empty() && running_) {
                auto response = std::move(response_queue_.front());
                response_queue_.pop();
                lock.unlock();
                
                // 发送响应
                ssize_t bytes_sent = write(response->client_fd, 
                                         response->data.c_str(), 
                                         response->data.length());
                
                if (bytes_sent < 0) {
                    perror("async_write failed");
                }
                
                close(response->client_fd);
                
                lock.lock();
            }
        }
    }
    
private:
    std::atomic<bool> running_;
    std::thread io_thread_;
    std::queue<std::unique_ptr<AsyncResponse>> response_queue_;
    std::mutex response_mutex_;
    std::condition_variable response_cv_;
};

// 同步处理层
class SyncProcessLayer {
public:
    SyncProcessLayer(int num_workers, AsyncIOLayer& async_layer) 
        : async_layer_(async_layer), running_(true) {
        
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&SyncProcessLayer::worker_loop, this, i);
        }
    }
    
    void start() {
        running_ = true;
    }
    
    void stop() {
        running_ = false;
        request_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    void process_request(std::unique_ptr<Request> request) {
        std::lock_guard<std::mutex> lock(request_mutex_);
        request_queue_.push(std::move(request));
        request_cv_.notify_one();
    }
    
private:
    void worker_loop(int worker_id) {
        std::cout << "Sync worker " << worker_id << " started" << std::endl;
        
        while (running_) {
            std::unique_lock<std::mutex> lock(request_mutex_);
            request_cv_.wait(lock, [this] { 
                return !request_queue_.empty() || !running_; 
            });
            
            if (!running_) break;
            
            if (!request_queue_.empty()) {
                auto request = std::move(request_queue_.front());
                request_queue_.pop();
                lock.unlock();
                
                // 同步处理请求
                process_request_sync(worker_id, std::move(request));
            }
        }
        
        std::cout << "Sync worker " << worker_id << " stopped" << std::endl;
    }
    
    void process_request_sync(int worker_id, std::unique_ptr<Request> request) {
        #ifdef DEBUG
        std::cout << "Worker " << worker_id << " processing request from client (fd: " 
                  << request->client_fd << "): " << request->data << std::endl;
        #endif
        
        // 模拟同步业务逻辑处理
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // 生成响应
        std::string response_data = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: 13\r\n"
            "\r\n"
            "Hello, World!";
        
        auto response = std::unique_ptr<AsyncResponse>(new AsyncResponse(request->client_fd, response_data));
        
        // 将响应交给异步I/O层发送
        async_layer_.async_write(std::move(response));
    }
    
private:
    AsyncIOLayer& async_layer_;
    std::atomic<bool> running_;
    std::vector<std::thread> workers_;
    std::queue<std::unique_ptr<Request>> request_queue_;
    std::mutex request_mutex_;
    std::condition_variable request_cv_;
};

// 半同步/半异步服务器
class HalfSyncAsyncServer : public ServerBase {
public:
    HalfSyncAsyncServer(int port = 8080, int num_workers = 4)
        : ServerBase(port), sync_layer_(num_workers, async_layer_) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 启动异步I/O层
        async_layer_.start();
        
        // 启动同步处理层
        sync_layer_.start();
        
        running_ = true;
        std::cout << "Half-Sync/Half-Async Server started on port " << port_ << std::endl;
        
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
            
            // 异步读取客户端数据，如果立即读取失败则重试
            bool read_success = false;
            int retry_count = 0;
            const int max_retries = 10;
            
            while (!read_success && retry_count < max_retries) {
                read_success = async_layer_.async_read(client_fd, client_addr, 
                    [this](std::unique_ptr<Request> request) {
                        // 将请求交给同步处理层
                        sync_layer_.process_request(std::move(request));
                    });
                
                if (!read_success) {
                    // 短暂等待后重试
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    retry_count++;
                }
            }
            
            if (!read_success) {
                std::cerr << "Failed to read from client after " << max_retries << " retries" << std::endl;
                close(client_fd);
            }
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        sync_layer_.stop();
        async_layer_.stop();
    }
    
private:
    AsyncIOLayer async_layer_;
    SyncProcessLayer sync_layer_;
};

#endif // HALF_SYNC_ASYNC_SERVER_H