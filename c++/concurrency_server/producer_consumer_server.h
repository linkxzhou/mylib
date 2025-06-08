#ifndef PRODUCER_CONSUMER_SERVER_H
#define PRODUCER_CONSUMER_SERVER_H

#include "./base/server_base.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>

// 客户端连接任务
struct ClientTask {
    int client_fd;
    struct sockaddr_in client_addr;
    
    ClientTask(int fd, const struct sockaddr_in& addr) 
        : client_fd(fd), client_addr(addr) {}
};

// 线程安全的任务队列
class TaskQueue {
public:
    TaskQueue(size_t max_size = 1000) : max_size_(max_size), shutdown_(false) {}
    
    bool push(ClientTask task) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 等待队列有空间
        not_full_.wait(lock, [this] { 
            return queue_.size() < max_size_ || shutdown_; 
        });
        
        if (shutdown_) {
            return false;
        }
        
        queue_.push(std::move(task));
        not_empty_.notify_one();
        return true;
    }
    
    bool pop(ClientTask& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 等待队列有任务
        not_empty_.wait(lock, [this] { 
            return !queue_.empty() || shutdown_; 
        });
        
        if (shutdown_ && queue_.empty()) {
            return false;
        }
        
        task = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<ClientTask> queue_;
    size_t max_size_;
    bool shutdown_;
};

// 消费者线程
class ConsumerThread {
public:
    ConsumerThread(int id, TaskQueue& task_queue) 
        : id_(id), task_queue_(task_queue), running_(false) {}
    
    void start() {
        running_ = true;
        thread_ = std::thread(&ConsumerThread::run, this);
    }
    
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
private:
    void run() {
        std::cout << "Consumer thread " << id_ << " started" << std::endl;
        
        while (running_) {
            ClientTask task(0, {});
            if (task_queue_.pop(task)) {
                handle_client(task);
            }
        }
        
        std::cout << "Consumer thread " << id_ << " stopped" << std::endl;
    }
    
    void handle_client(const ClientTask& task) {
        std::cout << "Consumer " << id_ << " handling client (fd: " 
                  << task.client_fd << ")" << std::endl;
        
        char buffer[1024] = {0};
        ssize_t bytes_read = read(task.client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            #if DEBUG
            std::cout << "Consumer " << id_ << " received: " << buffer << std::endl;
            #endif
            
            const char* response = 
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html\r\n"
                "Content-Length: 13\r\n"
                "\r\n"
                "Hello, World!";
            
            write(task.client_fd, response, strlen(response));
        }
        
        close(task.client_fd);
    }
    
private:
    int id_;
    TaskQueue& task_queue_;
    std::atomic<bool> running_;
    std::thread thread_;
};

// 生产者-消费者服务器
class ProducerConsumerServer : public ServerBase {
public:
    ProducerConsumerServer(int port = 8080, int num_consumers = 4, size_t queue_size = 100)
        : ServerBase(port), task_queue_(queue_size) {
        
        // 创建消费者线程
        for (int i = 0; i < num_consumers; ++i) {
            consumers_.emplace_back(std::unique_ptr<ConsumerThread>(new ConsumerThread(i, task_queue_)));
        }
    }
    
    ~ProducerConsumerServer() {
        stop();
    }
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 启动所有消费者线程
        for (auto& consumer : consumers_) {
            consumer->start();
        }
        
        running_ = true;
        std::cout << "Producer-Consumer Server started on port " << port_ 
                  << " with " << consumers_.size() << " consumers" << std::endl;
        
        // 生产者循环 - 接收连接并放入队列
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
            std::cout << "Producer accepted client: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) 
                      << " (queue size: " << task_queue_.size() << ")" << std::endl;
            #endif

            // 将任务放入队列
            ClientTask task(client_fd, client_addr);
            if (!task_queue_.push(std::move(task))) {
                std::cerr << "Failed to enqueue task, server shutting down" << std::endl;
                close(client_fd);
                break;
            }
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        
        // 关闭任务队列
        task_queue_.shutdown();
        
        // 停止所有消费者线程
        for (auto& consumer : consumers_) {
            consumer->stop();
        }
    }
    
private:
    TaskQueue task_queue_;
    std::vector<std::unique_ptr<ConsumerThread>> consumers_;
};

#endif // PRODUCER_CONSUMER_SERVER_H