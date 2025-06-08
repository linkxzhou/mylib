#ifndef SINGLE_PROCESS_SERVER_H
#define SINGLE_PROCESS_SERVER_H

#include "./base/server_base.h"

// 单进程服务器 - 串行处理客户端请求
class SingleProcessServer : public ServerBase {
public:
    SingleProcessServer(int port = 8080) : ServerBase(port) {}
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        running_ = true;
        std::cout << "Single Process Server started on port " << port_ << std::endl;
        
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                perror("accept failed");
                break;
            }
            
            std::cout << "Client connected: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) << std::endl;
            
            // 串行处理客户端请求
            handle_client(client_fd);
        }
        
        return true;
    }
};

#endif // SINGLE_PROCESS_SERVER_H