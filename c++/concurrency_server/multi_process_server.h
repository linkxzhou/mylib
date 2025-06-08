#ifndef MULTI_PROCESS_SERVER_H
#define MULTI_PROCESS_SERVER_H

#include "./base/server_base.h"

// 多进程服务器 - 为每个客户端fork一个新进程
class MultiProcessServer : public ServerBase {
public:
    MultiProcessServer(int port = 8080) : ServerBase(port) {
        // 设置信号处理，避免僵尸进程
        signal(SIGCHLD, SIG_IGN);
    }
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        running_ = true;
        std::cout << "Multi Process Server started on port " << port_ << std::endl;
        
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
            std::cout << "Multi Process Server: client_fd = " << client_fd << std::endl;
            std::cout << "Client connected: " << inet_ntoa(client_addr.sin_addr) 
                      << ":" << ntohs(client_addr.sin_port) << std::endl;
            #endif
            
            // Fork子进程处理客户端
            pid_t pid = fork();
            if (pid == 0) {
                // 子进程
                close(server_fd_);  // 子进程不需要监听socket
                handle_client(client_fd);
                exit(0);
            } else if (pid > 0) {
                // 父进程
                close(client_fd);   // 父进程不需要客户端socket
            } else {
                perror("fork failed");
                close(client_fd);
            }
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        // 等待所有子进程结束
        while (waitpid(-1, nullptr, WNOHANG) > 0) {
            // 回收子进程
        }
    }
};

#endif // MULTI_PROCESS_SERVER_H