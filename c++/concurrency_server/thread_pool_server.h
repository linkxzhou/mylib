#ifndef THREAD_POOL_SERVER_H
#define THREAD_POOL_SERVER_H

#include "./base/server_base.h"

// 线程池服务器 - 使用预先创建的线程池处理客户端请求
class ThreadPoolServer : public ServerBase {
public:
    ThreadPoolServer(int port = 8080, size_t pool_size = std::thread::hardware_concurrency()) 
        : ServerBase(port), thread_pool_(pool_size) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        running_ = true;
        std::cout << "Thread Pool Server started on port " << port_ << std::endl;
        
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
            
            // 将客户端处理任务提交给线程池
            thread_pool_.enqueue([this, client_fd]() {
                handle_client(client_fd);
            });
        }
        
        return true;
    }
    
private:
    ThreadPool thread_pool_;
};

#endif // THREAD_POOL_SERVER_H