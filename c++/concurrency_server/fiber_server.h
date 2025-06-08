#ifndef FIBER_SERVER_H
#define FIBER_SERVER_H

#include "./base/server_base.h"
#include <vector>
#include <memory>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// 简化的纤程状态
enum class FiberState {
    READY,
    RUNNING,
    SUSPENDED,
    FINISHED
};

// 纤程基类
class Fiber {
public:
    Fiber(int id, std::function<void()> task) 
        : id_(id), task_(task), state_(FiberState::READY) {}
    
    virtual ~Fiber() = default;
    
    void run() {
        if (state_ == FiberState::READY || state_ == FiberState::SUSPENDED) {
            state_ = FiberState::RUNNING;
            task_();
            state_ = FiberState::FINISHED;
        }
    }
    
    void suspend() {
        state_ = FiberState::SUSPENDED;
    }
    
    void resume() {
        if (state_ == FiberState::SUSPENDED) {
            state_ = FiberState::READY;
        }
    }
    
    FiberState get_state() const { return state_; }
    int get_id() const { return id_; }
    bool is_finished() const { return state_ == FiberState::FINISHED; }
    
private:
    int id_;
    std::function<void()> task_;
    FiberState state_;
};

// 客户端处理纤程
class ClientFiber : public Fiber {
public:
    ClientFiber(int id, int client_fd) 
        : Fiber(id, [this]() { handle_client(); }), client_fd_(client_fd) {}
    
    ~ClientFiber() {
        if (client_fd_ != -1) {
            close(client_fd_);
        }
    }
    
private:
    void handle_client() {
        char buffer[1024] = {0};
        ssize_t bytes_read = read(client_fd_, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            std::cout << "Fiber " << get_id() << " processing client (fd: " 
                      << client_fd_ << "): " << buffer << std::endl;
            
            const char* response = 
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html\r\n"
                "Content-Length: 13\r\n"
                "\r\n"
                "Hello, World!";
            
            write(client_fd_, response, strlen(response));
        }
        
        close(client_fd_);
        client_fd_ = -1;
    }
    
private:
    int client_fd_;
};

// 纤程调度器
class FiberScheduler {
public:
    FiberScheduler(int num_workers = 2) : running_(true), next_fiber_id_(0) {
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&FiberScheduler::worker_loop, this);
        }
    }
    
    ~FiberScheduler() {
        stop();
    }
    
    void start() {
        running_ = true;
    }
    
    void stop() {
        running_ = false;
        cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    void schedule_fiber(std::unique_ptr<Fiber> fiber) {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_queue_.push(std::move(fiber));
        cv_.notify_one();
    }
    
    void add_client_fiber(int client_fd) {
        auto fiber = std::unique_ptr<ClientFiber>(new ClientFiber(next_fiber_id_++, client_fd));
        schedule_fiber(std::move(fiber));
    }
    
private:
    void worker_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !ready_queue_.empty() || !running_; });
            
            if (!running_) break;
            
            if (!ready_queue_.empty()) {
                auto fiber = std::move(ready_queue_.front());
                ready_queue_.pop();
                lock.unlock();
                
                // 运行纤程
                fiber->run();
                
                // 纤程执行完毕，不需要重新调度
                // ClientFiber是一次性任务，执行完就结束
            }
        }
    }
    
private:
    std::atomic<bool> running_;
    std::atomic<int> next_fiber_id_;
    std::queue<std::unique_ptr<Fiber>> ready_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;
};

// 纤程服务器
class FiberServer : public ServerBase {
public:
    FiberServer(int port = 8080, int num_workers = 2) 
        : ServerBase(port), scheduler_(num_workers) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        scheduler_.start();
        running_ = true;
        std::cout << "Fiber Server started on port " << port_ << std::endl;
        
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
            
            // 创建纤程处理客户端
            scheduler_.add_client_fiber(client_fd);
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        scheduler_.stop();
    }
    
private:
    FiberScheduler scheduler_;
};

#endif // FIBER_SERVER_H