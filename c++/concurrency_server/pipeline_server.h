#ifndef PIPELINE_SERVER_H
#define PIPELINE_SERVER_H

#include "./base/server_base.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <memory>

// 管道阶段数据
struct PipelineData {
    int client_fd;
    struct sockaddr_in client_addr;
    std::string raw_data;
    std::string request_data;
    std::string processed_data;
    std::string response_data;
    int current_stage;
    
    PipelineData(int fd, const struct sockaddr_in& addr)
        : client_fd(fd), client_addr(addr), current_stage(0) {}
};

// 管道响应数据结构
struct PipelineResponse {
    int client_fd;
    std::string data;
    
    PipelineResponse(int fd, const std::string& response_data) 
        : client_fd(fd), data(response_data) {}
};

// 管道阶段基类
class PipelineStage {
public:
    PipelineStage(int stage_id, int num_workers) 
        : stage_id_(stage_id), running_(true) {
        
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&PipelineStage::worker_loop, this, i);
        }
    }
    
    virtual ~PipelineStage() {
        stop();
    }
    
    void start() {
        running_ = true;
    }
    
    void stop() {
        running_ = false;
        input_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    void add_data(std::shared_ptr<PipelineData> data) {
        std::lock_guard<std::mutex> lock(input_mutex_);
        input_queue_.push(data);
        input_cv_.notify_one();
    }
    
    void set_next_stage(std::shared_ptr<PipelineStage> next) {
        next_stage_ = next;
    }
    
    virtual void process_data(std::shared_ptr<PipelineData> data) = 0;
    
protected:
    void forward_to_next_stage(std::shared_ptr<PipelineData> data) {
        data->current_stage++;
        if (next_stage_) {
            next_stage_->add_data(data);
        }
    }
    
private:
    void worker_loop(int worker_id) {
        std::cout << "Stage " << stage_id_ << " worker " << worker_id << " started" << std::endl;
        
        while (running_) {
            std::unique_lock<std::mutex> lock(input_mutex_);
            input_cv_.wait(lock, [this] { 
                return !input_queue_.empty() || !running_; 
            });
            
            if (!running_) break;
            
            if (!input_queue_.empty()) {
                auto data = input_queue_.front();
                input_queue_.pop();
                lock.unlock();
                
                // 处理数据
                process_data(data);
            }
        }
        
        #if DEBUG
        std::cout << "Stage " << stage_id_ << " worker " << worker_id << " stopped" << std::endl;
        #endif
    }
    
protected:
    int stage_id_;
    std::atomic<bool> running_;
    std::shared_ptr<PipelineStage> next_stage_;
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::shared_ptr<PipelineData>> input_queue_;
    std::mutex input_mutex_;
    std::condition_variable input_cv_;
};

// 阶段1：数据接收和解析
class ReceiveStage : public PipelineStage {
public:
    ReceiveStage(int num_workers = 2) : PipelineStage(1, num_workers) {}
    
    void process_data(std::shared_ptr<PipelineData> data) override {
        std::cout << "Stage 1: Receiving data from client (fd: " 
                  << data->client_fd << ")" << std::endl;
        
        // 读取客户端数据
        char buffer[1024] = {0};
        ssize_t bytes_read = read(data->client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            data->raw_data = std::string(buffer, bytes_read);

            #if DEBUG
            std::cout << "Stage 1: Received " << bytes_read << " bytes" << std::endl;
            #endif

            // 转发到下一阶段
            forward_to_next_stage(data);
        } else {
            // 读取失败，关闭连接
            std::cerr << "Stage 1: Failed to read from client" << std::endl;
            close(data->client_fd);
        }
    }
};

// 阶段2：数据处理和业务逻辑
class ProcessStage : public PipelineStage {
public:
    ProcessStage(int num_workers = 4) : PipelineStage(2, num_workers) {}
    
    void process_data(std::shared_ptr<PipelineData> data) override {
        std::cout << "Stage 2: Processing data for client (fd: " 
                  << data->client_fd << ")" << std::endl;
        
        // 模拟业务逻辑处理
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // 处理原始数据
        data->processed_data = "Processed: " + data->raw_data;
        
        #if DEBUG
        std::cout << "Stage 2: Data processed" << std::endl;
        #endif

        // 转发到下一阶段
        forward_to_next_stage(data);
    }
};

// 阶段3：响应生成
class ResponseStage : public PipelineStage {
public:
    ResponseStage(int num_workers = 2) : PipelineStage(3, num_workers) {}
    
    void process_data(std::shared_ptr<PipelineData> data) override {
        #if DEBUG
        std::cout << "Stage 3: Generating response for client (fd: " 
                  << data->client_fd << ")" << std::endl;
        #endif
        
        // 生成HTTP响应
        data->response_data = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: 13\r\n"
            "\r\n"
            "Hello, World!";
        
        #if DEBUG
        std::cout << "Stage 3: Response generated" << std::endl;
        #endif

        // 转发到下一阶段
        forward_to_next_stage(data);
    }
};

// 阶段4：数据发送
class SendStage : public PipelineStage {
public:
    SendStage(int num_workers = 2) : PipelineStage(4, num_workers) {}
    
    void process_data(std::shared_ptr<PipelineData> data) override {
        #if DEBUG
        std::cout << "Stage 4: Sending response to client (fd: " 
                  << data->client_fd << ")" << std::endl;
        #endif
        
        // 发送响应
        ssize_t bytes_sent = write(data->client_fd, 
                                 data->response_data.c_str(), 
                                 data->response_data.length());
        
        #if DEBUG
        if (bytes_sent > 0) {
            std::cout << "Stage 4: Sent " << bytes_sent << " bytes" << std::endl;
        } else {
            std::cerr << "Stage 4: Failed to send response" << std::endl;
        }
        #endif

        // 关闭连接
        close(data->client_fd);

        #if DEBUG
        std::cout << "Stage 4: Connection closed" << std::endl;
        #endif
    }
};

// 管道服务器
class PipelineServer : public ServerBase {
public:
    PipelineServer(int port = 8080) : ServerBase(port) {
        // 创建管道阶段
        receive_stage_ = std::make_shared<ReceiveStage>(2);
        process_stage_ = std::make_shared<ProcessStage>(4);
        response_stage_ = std::make_shared<ResponseStage>(2);
        send_stage_ = std::make_shared<SendStage>(2);
        
        // 连接管道阶段
        receive_stage_->set_next_stage(process_stage_);
        process_stage_->set_next_stage(response_stage_);
        response_stage_->set_next_stage(send_stage_);
    }
    
    bool start() override {
        if (!create_listen_socket()) {
            return false;
        }
        
        // 启动所有管道阶段
        receive_stage_->start();
        process_stage_->start();
        response_stage_->start();
        send_stage_->start();
        
        running_ = true;
        std::cout << "Pipeline Server started on port " << port_ << std::endl;
        
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

            // 创建管道数据并送入第一阶段
            auto pipeline_data = std::make_shared<PipelineData>(client_fd, client_addr);
            receive_stage_->add_data(pipeline_data);
        }
        
        return true;
    }
    
    void stop() override {
        ServerBase::stop();
        
        // 停止所有管道阶段
        send_stage_->stop();
        response_stage_->stop();
        process_stage_->stop();
        receive_stage_->stop();
    }
    
private:
    std::shared_ptr<ReceiveStage> receive_stage_;
    std::shared_ptr<ProcessStage> process_stage_;
    std::shared_ptr<ResponseStage> response_stage_;
    std::shared_ptr<SendStage> send_stage_;
};

#endif // PIPELINE_SERVER_H