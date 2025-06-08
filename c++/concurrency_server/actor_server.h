#ifndef ACTOR_SERVER_H
#define ACTOR_SERVER_H

#include "./base/server_base.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <atomic>

// 消息基类
class Message {
public:
    virtual ~Message() = default;
    virtual void process() = 0;
};

// 客户端消息
class ClientMessage : public Message {
public:
    ClientMessage(int client_fd) : client_fd_(client_fd) {}
    
    void process() override {
        char buffer[1024] = {0};
        ssize_t bytes_read = read(client_fd_, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            std::cout << "Actor processing message from client (fd: " << client_fd_ 
                      << "): " << buffer << std::endl;
            
            const char* response = 
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html\r\n"
                "Content-Length: 13\r\n"
                "\r\n"
                "Hello, World!";
            
            write(client_fd_, response, strlen(response));
        }
        
        close(client_fd_);
    }
    
private:
    int client_fd_;
};

// Actor基类
class Actor {
public:
    Actor(int id) : id_(id), running_(false) {}
    virtual ~Actor() = default;
    
    void start() {
        running_ = true;
        worker_thread_ = std::thread(&Actor::run, this);
    }
    
    void stop() {
        running_ = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    void send_message(std::unique_ptr<Message> message) {
        std::lock_guard<std::mutex> lock(mutex_);
        message_queue_.push(std::move(message));
        cv_.notify_one();
    }
    
    int get_id() const { return id_; }
    
private:
    void run() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !message_queue_.empty() || !running_; });
            
            while (!message_queue_.empty() && running_) {
                auto message = std::move(message_queue_.front());
                message_queue_.pop();
                lock.unlock();
                
                message->process();
                
                lock.lock();
            }
        }
    }
    
private:
    int id_;
    std::atomic<bool> running_;
    std::queue<std::unique_ptr<Message>> message_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
};

// Actor系统管理器
class ActorSystem {
public:
    ActorSystem(int num_actors = 4) {
        for (int i = 0; i < num_actors; ++i) {
            auto actor = std::unique_ptr<Actor>(new Actor(i));
            actor->start();
            actors_.push_back(std::move(actor));
        }
    }
    
    ~ActorSystem() {
        for (auto& actor : actors_) {
            actor->stop();
        }
    }
    
    void dispatch_message(std::unique_ptr<Message> message) {
        // 简单的轮询调度
        int actor_id = next_actor_++ % actors_.size();
        actors_[actor_id]->send_message(std::move(message));
    }
    
private:
    std::vector<std::unique_ptr<Actor>> actors_;
    std::atomic<int> next_actor_{0};
};

// Actor模型服务器
class ActorServer : public ServerBase {
public:
    ActorServer(int port = 8080, int num_actors = 4) 
        : ServerBase(port), actor_system_(num_actors) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        running_ = true;
        std::cout << "Actor Server started on port " << port_ << std::endl;
        
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
            
            // 创建消息并分发给Actor
            auto message = std::unique_ptr<ClientMessage>(new ClientMessage(client_fd));
            actor_system_.dispatch_message(std::move(message));
        }
        
        return true;
    }
    
private:
    ActorSystem actor_system_;
};

#endif // ACTOR_SERVER_H