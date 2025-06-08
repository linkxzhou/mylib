#ifndef PROACTOR_SERVER_H
#define PROACTOR_SERVER_H

#include "./base/server_base.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <memory>
#include <functional>

// 异步操作类型
enum class AsyncOpType {
    READ,
    WRITE,
    ACCEPT
};

// 异步操作完成事件
struct CompletionEvent {
    AsyncOpType op_type;
    int fd;
    std::vector<char> buffer;
    ssize_t bytes_transferred;
    int error_code;
    std::function<void(const CompletionEvent&)> completion_handler;
    
    CompletionEvent(AsyncOpType type, int file_desc, 
                   std::function<void(const CompletionEvent&)> handler)
        : op_type(type), fd(file_desc), bytes_transferred(0), 
          error_code(0), completion_handler(handler) {
        buffer.resize(1024);
    }
};

// 异步I/O操作发起器
class AsyncInitiator {
public:
    AsyncInitiator() : running_(false) {}
    
    void start() {
        running_ = true;
        io_thread_ = std::thread(&AsyncInitiator::io_loop, this);
    }
    
    void stop() {
        running_ = false;
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }
    
    // 发起异步读操作
    void async_read(int fd, std::function<void(const CompletionEvent&)> handler) {
        auto event = std::make_shared<CompletionEvent>(AsyncOpType::READ, fd, handler);
        
        // 在后台线程中执行I/O操作
        std::thread([this, event]() {
            // 模拟异步读取
            ssize_t bytes_read = read(event->fd, event->buffer.data(), event->buffer.size() - 1);
            
            event->bytes_transferred = bytes_read;
            if (bytes_read < 0) {
                event->error_code = errno;
            } else {
                event->buffer[bytes_read] = '\0';
            }
            
            // 将完成事件放入完成队列
            post_completion(event);
        }).detach();
    }
    
    // 发起异步写操作
    void async_write(int fd, const std::string& data, 
                    std::function<void(const CompletionEvent&)> handler) {
        auto event = std::make_shared<CompletionEvent>(AsyncOpType::WRITE, fd, handler);
        
        // 在后台线程中执行I/O操作
        std::thread([this, event, data]() {
            // 模拟异步写入
            ssize_t bytes_written = write(event->fd, data.c_str(), data.length());
            
            event->bytes_transferred = bytes_written;
            if (bytes_written < 0) {
                event->error_code = errno;
            }
            
            // 将完成事件放入完成队列
            post_completion(event);
        }).detach();
    }
    
    // 获取完成事件
    bool get_completion_event(std::shared_ptr<CompletionEvent>& event) {
        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] { 
            return !completion_queue_.empty() || !running_; 
        });
        
        if (!running_ && completion_queue_.empty()) {
            return false;
        }
        
        if (!completion_queue_.empty()) {
            event = completion_queue_.front();
            completion_queue_.pop();
            return true;
        }
        
        return false;
    }
    
private:
    void io_loop() {
        // I/O线程主要用于管理异步操作
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void post_completion(std::shared_ptr<CompletionEvent> event) {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        completion_queue_.push(event);
        completion_cv_.notify_one();
    }
    
private:
    std::atomic<bool> running_;
    std::thread io_thread_;
    std::queue<std::shared_ptr<CompletionEvent>> completion_queue_;
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
};

// 完成处理器
class CompletionHandler {
public:
    virtual ~CompletionHandler() = default;
    virtual void handle_completion(const CompletionEvent& event) = 0;
};

// 读完成处理器
class ReadCompletionHandler : public CompletionHandler {
public:
    ReadCompletionHandler(AsyncInitiator& initiator) : initiator_(initiator) {}
    
    void handle_completion(const CompletionEvent& event) override {
        if (event.error_code != 0) {
            std::cerr << "Read error: " << strerror(event.error_code) << std::endl;
            close(event.fd);
            return;
        }
        
        if (event.bytes_transferred > 0) {
            #if DEBUG
            std::cout << "Read completed for fd " << event.fd 
                      << ", bytes: " << event.bytes_transferred << std::endl;
            #endif
            
            // 处理读取的数据
            std::string request(event.buffer.data(), event.bytes_transferred);
            #if DEBUG
            std::cout << "Received: " << request << std::endl;
            #endif
            
            // 发起异步写操作发送响应
            std::string response = 
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html\r\n"
                "Content-Length: 13\r\n"
                "\r\n"
                "Hello, World!";
            
            initiator_.async_write(event.fd, response, 
                [](const CompletionEvent& write_event) {

                    #if DEBUG
                    if (write_event.error_code == 0) {
                        std::cout << "Write completed for fd " << write_event.fd 
                                  << ", bytes: " << write_event.bytes_transferred << std::endl;
                    } else {
                        std::cerr << "Write error: " << strerror(write_event.error_code) << std::endl;
                    }
                    #endif

                    close(write_event.fd);
                });
        } else {
            // 客户端关闭连接
            std::cout << "Client disconnected (fd: " << event.fd << ")" << std::endl;
            close(event.fd);
        }
    }
    
private:
    AsyncInitiator& initiator_;
};

// Proactor服务器
class ProactorServer : public ServerBase {
public:
    ProactorServer(int port = 8080) 
        : ServerBase(port), read_handler_(initiator_) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 启动异步I/O发起器
        initiator_.start();
        
        // 启动完成事件处理线程
        completion_thread_ = std::thread(&ProactorServer::completion_loop, this);
        
        running_ = true;
        std::cout << "Proactor Server started on port " << port_ << std::endl;
        
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                perror("accept failed");
                break;
            }
            
            #if DEBUG
            std::cout << "Client connected: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) << std::endl;
            #endif
            
            // 发起异步读操作
            initiator_.async_read(client_fd, 
                [this](const CompletionEvent& event) {
                    read_handler_.handle_completion(event);
                });
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        initiator_.stop();
        
        if (completion_thread_.joinable()) {
            completion_thread_.join();
        }
    }
    
private:
    void completion_loop() {
        while (running_) {
            std::shared_ptr<CompletionEvent> event;
            if (initiator_.get_completion_event(event)) {
                // 调用相应的完成处理器
                if (event->completion_handler) {
                    event->completion_handler(*event);
                }
            }
        }
    }
    
private:
    AsyncInitiator initiator_;
    ReadCompletionHandler read_handler_;
    std::thread completion_thread_;
};

#endif // PROACTOR_SERVER_H