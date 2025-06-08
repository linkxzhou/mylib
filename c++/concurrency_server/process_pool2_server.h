#ifndef PROCESS_POOL2_SERVER_H
#define PROCESS_POOL2_SERVER_H

#include "./base/server_base.h"

// 进程池服务器2 - 使用SO_REUSEPORT，每个进程独立监听
class ProcessPool2Server : public ServerBase {
public:
    ProcessPool2Server(int port = 8080, int pool_size = 4) 
        : ServerBase(port), pool_size_(pool_size) {
        signal(SIGCHLD, SIG_IGN);
    }
    
    bool start() override {
        running_ = true;
        std::cout << "Process Pool2 Server (SO_REUSEPORT) started on port " << port_ 
                  << " with " << pool_size_ << " worker processes" << std::endl;
        
        // 创建工作进程池
        for (int i = 0; i < pool_size_; ++i) {
            pid_t pid = fork();
            if (pid == 0) {
                // 子进程 - 工作进程
                worker_process();
                exit(0);
            } else if (pid > 0) {
                worker_pids_.push_back(pid);
            } else {
                perror("fork failed");
                return false;
            }
        }
        
        // 主进程等待
        while (running_) {
            sleep(1);
        }
        
        return true;
    }
    
    void stop() override {
        running_ = false;
        
        // 终止所有工作进程
        for (pid_t pid : worker_pids_) {
            kill(pid, SIGTERM);
        }
        
        // 等待所有工作进程结束
        for (pid_t pid : worker_pids_) {
            waitpid(pid, nullptr, 0);
        }
        
        ServerBase::stop();
    }
    
private:
    bool create_reuseport_socket() {
        int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_fd == -1) {
            perror("socket creation failed");
            return false;
        }
        
        int opt = 1;
        // 设置SO_REUSEADDR
        if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            perror("setsockopt SO_REUSEADDR failed");
            close(sock_fd);
            return false;
        }
        
        // 设置SO_REUSEPORT（Linux特性）
#ifdef SO_REUSEPORT
        if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
            perror("setsockopt SO_REUSEPORT failed");
            close(sock_fd);
            return false;
        }
#else
        std::cout << "SO_REUSEPORT not supported on this system" << std::endl;
#endif
        
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port_);
        
        if (bind(sock_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            perror("bind failed");
            close(sock_fd);
            return false;
        }
        
        if (listen(sock_fd, 10) < 0) {
            perror("listen failed");
            close(sock_fd);
            return false;
        }
        
        server_fd_ = sock_fd;
        return true;
    }
    
    void worker_process() {
        // 每个工作进程创建自己的监听socket
        if (!create_reuseport_socket()) {
            std::cerr << "Worker process " << getpid() << " failed to create socket" << std::endl;
            return;
        }
        
        std::cout << "Worker process " << getpid() << " started" << std::endl;
        
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                perror("accept failed in worker");
                break;
            }
            
            #if DEBUG
            std::cout << "Worker process " << getpid() << " handling client: " 
                      << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << std::endl;
            #endif
            
            handle_client(client_fd);
        }
        
        close(server_fd_);
    }
    
private:
    int pool_size_;
    std::vector<pid_t> worker_pids_;
};

#endif // PROCESS_POOL2_SERVER_H