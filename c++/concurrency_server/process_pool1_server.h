#ifndef PROCESS_POOL1_SERVER_H
#define PROCESS_POOL1_SERVER_H

#include "./base/server_base.h"

// 进程池服务器1 - 预先创建固定数量的工作进程
class ProcessPool1Server : public ServerBase {
public:
    ProcessPool1Server(int port = 8080, int pool_size = 4) 
        : ServerBase(port), pool_size_(pool_size) {
        signal(SIGCHLD, SIG_IGN);
    }
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        running_ = true;

        std::cout << "Process Pool1 Server started on port " << port_ 
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
    void worker_process() {
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            // 多个进程竞争accept同一个监听socket
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
    }
    
private:
    int pool_size_;
    std::vector<pid_t> worker_pids_;
};

#endif // PROCESS_POOL1_SERVER_H