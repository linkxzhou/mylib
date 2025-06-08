#ifndef LEADER_FOLLOWER_SERVER_H
#define LEADER_FOLLOWER_SERVER_H

#include "./base/server_base.h"

// 领导者/跟随者服务器 - 线程池的变种，一个线程作为leader监听连接
class LeaderFollowerServer : public ServerBase {
public:
    LeaderFollowerServer(int port = 8080, size_t pool_size = std::thread::hardware_concurrency()) 
        : ServerBase(port), pool_size_(pool_size), leader_index_(0) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        running_ = true;
        std::cout << "Leader-Follower Server started on port " << port_ 
                  << " with " << pool_size_ << " threads" << std::endl;
        
        // 创建线程池
        for (size_t i = 0; i < pool_size_; ++i) {
            threads_.emplace_back(&LeaderFollowerServer::worker_thread, this, i);
        }
        
        // 等待所有线程结束
        for (auto& thread : threads_) {
            thread.join();
        }
        
        return true;
    }
    
    void stop() override {
        running_ = false;
        // 唤醒所有等待的线程
        leader_cv_.notify_all();
        ServerBase::stop();
    }
    
private:
    void worker_thread(size_t thread_id) {
        std::cout << "Worker thread " << thread_id << " started" << std::endl;
        
        while (running_) {
            // 竞争成为leader
            std::unique_lock<std::mutex> lock(leader_mutex_);
            
            // 等待成为leader的机会
            leader_cv_.wait(lock, [this, thread_id] {
                return !running_ || leader_index_ == thread_id;
            });
            
            if (!running_) break;
            
            // 现在是leader，监听客户端连接
            lock.unlock();
            
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EINTR || errno == EAGAIN) {
                    // 重新竞争leader
                    promote_new_leader();
                    continue;
                }
                perror("accept failed");
                break;
            }
            
            #if DEBUG
            std::cout << "Thread " << thread_id << " (leader) accepted client: " 
                      << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << std::endl;
            #endif
            
            // 放弃leader身份，提升新的leader
            promote_new_leader();
            
            // 处理客户端请求（现在是worker）
            #if DEBUG
            std::cout << "Thread " << thread_id << " (worker) handling client" << std::endl;
            #endif
            
            handle_client(client_fd);
        }
        
        std::cout << "Worker thread " << thread_id << " finished" << std::endl;
    }
    
    void promote_new_leader() {
        std::lock_guard<std::mutex> lock(leader_mutex_);
        leader_index_ = (leader_index_ + 1) % pool_size_;
        leader_cv_.notify_all();
    }
    
private:
    size_t pool_size_;
    std::vector<std::thread> threads_;
    std::mutex leader_mutex_;
    std::condition_variable leader_cv_;
    size_t leader_index_;  // 当前leader线程的索引
};

#endif // LEADER_FOLLOWER_SERVER_H