#ifndef MULTI_THREAD_SERVER_H
#define MULTI_THREAD_SERVER_H

#include "./base/server_base.h"

// 多线程服务器 - 为每个客户端创建一个新线程
class MultiThreadServer : public ServerBase {
public:
    MultiThreadServer(int port = 8080) : ServerBase(port) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        running_ = true;
        std::cout << "Multi Thread Server started on port " << port_ << std::endl;
        
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
            
            // 创建新线程处理客户端
            std::thread client_thread([this, client_fd]() {
                handle_client(client_fd);
            });
            
            // 分离线程，让其自动清理
            client_thread.detach();
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        // 注意：这里没有等待所有线程结束的机制
        // 在实际应用中可能需要维护线程列表并等待它们结束
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
};

#endif // MULTI_THREAD_SERVER_H